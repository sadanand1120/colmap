// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/view_graph_calibration.h"

#include "colmap/estimators/cost_functions/calibration.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace colmap {
namespace {

// Lower bound for focal length optimization to prevent numerical issues.
constexpr double kFocalLengthLowerBound = 1e-3;

class ViewGraphCalibrationProgress {
 public:
  explicit ViewGraphCalibrationProgress(const size_t total_stages)
      : total_stages_(std::max<size_t>(total_stages, 1)),
#ifdef _WIN32
        enabled_(false),
#else
        enabled_(isatty(fileno(stderr))),
#endif
        output_fd_(-1),
        finished_(false),
        rendered_lines_(0) {
    timer_.Start();
#ifndef _WIN32
    if (enabled_) {
      output_fd_ = dup(fileno(stderr));
    }
#endif
    if (IsEnabled()) {
      heartbeat_thread_ = std::thread([this]() { HeartbeatLoop(); });
    }
  }

  ~ViewGraphCalibrationProgress() {
    StopHeartbeat();
    std::lock_guard<std::mutex> lock(mutex_);
    ClearLocked();
#ifndef _WIN32
    if (output_fd_ != -1) {
      close(output_fd_);
      output_fd_ = -1;
    }
#endif
  }

  void SetStage(const size_t stage_index, std::string stage) {
    std::lock_guard<std::mutex> lock(mutex_);
    stage_index_ = std::min(stage_index, total_stages_);
    stage_ = std::move(stage);
    has_bounded_work_ = false;
    RenderLocked(true);
  }

  void SetBoundedWork(std::string label,
                      const size_t current,
                      const size_t total) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total == 0) {
      has_bounded_work_ = false;
      RenderLocked();
      return;
    }
    const bool start_new_work = !has_bounded_work_ || bounded_label_ != label ||
                                bounded_total_ != total;
    bounded_label_ = std::move(label);
    bounded_current_ = std::min(current, total);
    bounded_total_ = total;
    has_bounded_work_ = true;
    RenderLocked(start_new_work || bounded_current_ == bounded_total_);
  }

  void ClearBoundedWork() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_bounded_work_) {
      return;
    }
    has_bounded_work_ = false;
    RenderLocked(true);
  }

  void Finish(const std::string& stage) {
    StopHeartbeat();
    std::string elapsed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (finished_) {
        return;
      }
      stage_index_ = total_stages_;
      stage_ = stage;
      has_bounded_work_ = false;
      RenderLocked(true);
      ClearLocked();
      elapsed = FormatElapsed();
      finished_ = true;
    }
    if (IsEnabled()) {
      Write(StringPrintf("View graph calibrator: %s (%zu/%zu stages, %s)\n",
                         stage.c_str(),
                         total_stages_,
                         total_stages_,
                         elapsed.c_str()));
    }
  }

 private:
  static std::string MakeBar(const size_t current,
                             const size_t total,
                             const size_t width) {
    if (total == 0) {
      return std::string(width, '-');
    }
    const size_t filled = std::min(width, width * current / total);
    return std::string(filled, '=') + std::string(width - filled, '-');
  }

  static std::string FormatPercent(const size_t current, const size_t total) {
    if (total == 0) {
      return "  0.0%";
    }
    return StringPrintf(
        "%5.1f%%",
        100.0 * static_cast<double>(current) / static_cast<double>(total));
  }

  std::string FormatElapsed() const {
    const auto total_seconds =
        static_cast<long long>(std::llround(timer_.ElapsedSeconds()));
    const auto hours = total_seconds / 3600;
    const auto minutes = (total_seconds % 3600) / 60;
    const auto seconds = total_seconds % 60;
    return StringPrintf("%lldh %02lldm %02llds", hours, minutes, seconds);
  }

  void HeartbeatLoop() {
    while (!heartbeat_stop_.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(kHeartbeatInterval);
      if (heartbeat_stop_.load(std::memory_order_relaxed)) {
        break;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      if (!finished_ && rendered_lines_ > 0) {
        RenderLocked(true);
      }
    }
  }

  void StopHeartbeat() {
    heartbeat_stop_.store(true, std::memory_order_relaxed);
    if (heartbeat_thread_.joinable()) {
      heartbeat_thread_.join();
    }
  }

  bool IsEnabled() const { return enabled_ && output_fd_ != -1; }

  void Write(const std::string& text) {
#ifndef _WIN32
    if (output_fd_ != -1) {
      const ssize_t num_bytes = write(output_fd_, text.data(), text.size());
      (void)num_bytes;
    }
#endif
  }

  void ClearLocked() {
    if (!IsEnabled() || rendered_lines_ == 0) {
      rendered_lines_ = 0;
      return;
    }
    Write("\r\033[2K");
    for (size_t i = 1; i < rendered_lines_; ++i) {
      Write("\033[1A\r\033[2K");
    }
    rendered_lines_ = 0;
    last_render_at_ = std::chrono::steady_clock::time_point::min();
  }

  void RenderLocked(const bool force = false) {
    if (!IsEnabled() || finished_) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!force && rendered_lines_ > 0 &&
        now - last_render_at_ < kMinRenderInterval) {
      return;
    }

    std::ostringstream line1;
    line1 << "View graph calibrator " << stage_index_ << "/" << total_stages_
          << " [" << MakeBar(stage_index_, total_stages_, 20) << "] "
          << FormatPercent(stage_index_, total_stages_) << " | " << stage_
          << " | " << FormatElapsed();

    std::ostringstream line2;
    if (has_bounded_work_) {
      line2 << "  " << bounded_current_ << "/" << bounded_total_ << " ["
            << MakeBar(bounded_current_, bounded_total_, 20) << "] "
            << FormatPercent(bounded_current_, bounded_total_) << " | "
            << bounded_label_;
    }

    ClearLocked();
    Write(FitLine(line1.str()));
    rendered_lines_ = 1;
    if (has_bounded_work_) {
      Write("\n" + FitLine(line2.str()));
      rendered_lines_ = 2;
    }
    last_render_at_ = now;
  }

  size_t TerminalColumns() const {
#ifndef _WIN32
    struct winsize size;
    if (output_fd_ != -1 && ioctl(output_fd_, TIOCGWINSZ, &size) == 0 &&
        size.ws_col > 0) {
      return std::max<size_t>(size.ws_col, 20);
    }
#endif
    return 80;
  }

  std::string FitLine(const std::string& line) const {
    const size_t columns = TerminalColumns();
    if (columns <= 1 || line.size() < columns) {
      return line;
    }
    return line.substr(0, columns - 1);
  }

  static constexpr auto kMinRenderInterval = std::chrono::milliseconds(120);
  static constexpr auto kHeartbeatInterval = std::chrono::seconds(1);

  Timer timer_;
  const size_t total_stages_;
  const bool enabled_;
  int output_fd_;
  std::atomic<bool> heartbeat_stop_{false};
  std::thread heartbeat_thread_;
  std::mutex mutex_;
  bool finished_;
  size_t rendered_lines_;
  std::chrono::steady_clock::time_point last_render_at_ =
      std::chrono::steady_clock::time_point::min();
  size_t stage_index_ = 0;
  std::string stage_ = "Starting";
  bool has_bounded_work_ = false;
  std::string bounded_label_;
  size_t bounded_current_ = 0;
  size_t bounded_total_ = 0;
};

void ReportProgress(
    const ViewGraphCalibrationOptions::ProgressCallback& progress,
    const std::string& label,
    const size_t current,
    const size_t total) {
  if (progress) {
    progress(label, current, total);
  }
}

class ViewGraphCalibrationCeresProgressCallback
    : public ceres::IterationCallback {
 public:
  ViewGraphCalibrationCeresProgressCallback(
      ViewGraphCalibrationOptions::ProgressCallback progress,
      std::string label,
      const int max_num_iterations)
      : progress_(std::move(progress)),
        label_(std::move(label)),
        max_num_iterations_(std::max(max_num_iterations, 1)) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    ReportProgress(progress_,
                   label_,
                   std::min(summary.iteration, max_num_iterations_),
                   max_num_iterations_);
    return ceres::SOLVER_CONTINUE;
  }

 private:
  ViewGraphCalibrationOptions::ProgressCallback progress_;
  const std::string label_;
  const int max_num_iterations_;
};

// Input for focal length calibration: an image pair with its F matrix.
struct FocalLengthCalibInput {
  image_pair_t pair_id = kInvalidImagePairId;
  camera_t camera_id1 = kInvalidCameraId;
  camera_t camera_id2 = kInvalidCameraId;
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
};

// Result of focal length calibration.
struct FocalLengthCalibResult {
  // Optimized focal lengths per camera.
  std::unordered_map<camera_t, double> focal_lengths;
  // Squared calibration error per image pair (unitless, relative error).
  std::unordered_map<image_pair_t, double> calibration_errors_sq;
  // Whether the calibration succeeded.
  bool success = false;
};

// Cross-validate prior focal lengths by checking the ratio of calibrated vs
// uncalibrated pairs per camera. UNCALIBRATED pairs are converted to
// CALIBRATED if both cameras have reliable priors.
size_t CrossValidatePriorFocalLengths(
    double min_calibrated_pair_ratio,
    const std::unordered_map<image_t, const Camera*>& image_id_to_camera,
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs,
    const ViewGraphCalibrationOptions::ProgressCallback& progress) {
  // For each camera, count the number of calibrated vs uncalibrated pairs.
  // first: total count, second: calibrated count.
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto& [pair_id, tvg] = pairs[i];
    ReportProgress(
        progress, "Counting calibrated pair ratios", i + 1, pairs.size());
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length) {
      continue;
    }

    auto& [total1, calibrated1] = camera_counter[camera1.camera_id];
    auto& [total2, calibrated2] = camera_counter[camera2.camera_id];
    ++total1;
    ++total2;
    if (tvg.config == TwoViewGeometry::CALIBRATED) {
      ++calibrated1;
      ++calibrated2;
    }
  }

  // Camera is valid if the ratio of calibrated pairs exceeds threshold.
  std::unordered_map<camera_t, bool> camera_validity;
  camera_validity.reserve(camera_counter.size());
  size_t camera_idx = 0;
  for (const auto& [camera_id, counter] : camera_counter) {
    ReportProgress(progress,
                   "Validating prior focal lengths",
                   ++camera_idx,
                   camera_counter.size());
    const auto [total, calibrated] = counter;
    const double ratio = static_cast<double>(calibrated) / total;
    camera_validity[camera_id] = ratio > min_calibrated_pair_ratio;
  }

  // Convert UNCALIBRATED pairs to CALIBRATED if both cameras are valid.
  // Compute E from F using the prior camera calibrations.
  size_t num_upgraded_pairs = 0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    auto& [pair_id, tvg] = pairs[i];
    ReportProgress(progress, "Upgrading reliable priors", i + 1, pairs.size());
    if (tvg.config != TwoViewGeometry::UNCALIBRATED) {
      continue;
    }

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);

    if (camera_validity[camera1.camera_id] &&
        camera_validity[camera2.camera_id]) {
      THROW_CHECK(tvg.F.has_value())
          << "UNCALIBRATED two-view geometry must have F matrix";
      tvg.E = EssentialFromFundamentalMatrix(camera2.CalibrationMatrix(),
                                             tvg.F.value(),
                                             camera1.CalibrationMatrix());
      tvg.config = TwoViewGeometry::CALIBRATED;
      ++num_upgraded_pairs;
    }
  }

  VLOG(1) << "Upgraded " << num_upgraded_pairs << " / " << pairs.size()
          << " pairs to calibrated through cross-validation";
  return num_upgraded_pairs;
}

// Re-estimate relative poses for all pairs using calibrated cameras.
void ReestimateRelativePoses(
    const ViewGraphCalibrationOptions& options,
    std::vector<std::pair<image_pair_t, TwoViewGeometry>>& pairs,
    const std::unordered_map<image_t, const Camera*>& image_id_to_camera,
    Database* database) {
  VLOG(1) << "Re-estimating relative poses for " << pairs.size() << " pairs";

  TwoViewGeometryOptions two_view_options;
  two_view_options.compute_relative_pose = true;
  two_view_options.ransac_options.max_error = options.relpose_max_error;
  two_view_options.min_num_inliers = options.relpose_min_num_inliers;
  two_view_options.min_inlier_ratio = options.relpose_min_inlier_ratio;
  two_view_options.ransac_options.random_seed = options.random_seed;

  // Pre-read all keypoints from database.
  std::unordered_set<image_t> image_ids;
  image_ids.reserve(2 * pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto& [pair_id, tvg] = pairs[i];
    ReportProgress(
        options.progress_callback, "Collecting image ids", i + 1, pairs.size());
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
  }

  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_points;
  image_points.reserve(image_ids.size());
  size_t image_idx = 0;
  for (const image_t image_id : image_ids) {
    ReportProgress(options.progress_callback,
                   "Reading keypoints",
                   ++image_idx,
                   image_ids.size());
    const FeatureKeypoints keypoints = database->ReadKeypoints(image_id);
    std::vector<Eigen::Vector2d> points(keypoints.size());
    for (size_t j = 0; j < keypoints.size(); ++j) {
      points[j] = Eigen::Vector2d(keypoints[j].x, keypoints[j].y);
    }
    image_points[image_id] = std::move(points);
  }

  // Parallel estimation with mutex-protected database access for matches.
  std::mutex database_mutex;
  std::atomic<size_t> num_processed_pairs{0};
  ThreadPool thread_pool(options.solver_options.num_threads);
  ReportProgress(options.progress_callback,
                 "Re-estimating relative poses",
                 0,
                 pairs.size());

  for (size_t i = 0; i < pairs.size(); ++i) {
    thread_pool.AddTask([&, i]() {
      auto& [pair_id, tvg] = pairs[i];
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

      FeatureMatches matches;
      {
        std::lock_guard<std::mutex> lock(database_mutex);
        matches = database->ReadMatches(image_id1, image_id2);
      }

      const Camera& camera1 = *image_id_to_camera.at(image_id1);
      const Camera& camera2 = *image_id_to_camera.at(image_id2);

      const std::vector<Eigen::Vector2d>& points1 = image_points.at(image_id1);
      const std::vector<Eigen::Vector2d>& points2 = image_points.at(image_id2);

      tvg = EstimateCalibratedTwoViewGeometry(
          camera1, points1, camera2, points2, matches, two_view_options);

      ReportProgress(options.progress_callback,
                     "Re-estimating relative poses",
                     num_processed_pairs.fetch_add(1) + 1,
                     pairs.size());
    });
  }
  thread_pool.Wait();
}

// Core Ceres optimization for focal length calibration.
// This is a pure function with no I/O dependencies.
// See: "Stable Intrinsic Auto-Calibration from Fundamental Matrices of Devices
// with Uncorrelated Camera Parameters", Fetzer et al., WACV 2020.
FocalLengthCalibResult CalibrateFocalLengths(
    const ViewGraphCalibrationOptions& options,
    const std::vector<FocalLengthCalibInput>& inputs,
    const std::unordered_map<camera_t, Camera>& cameras) {
  FocalLengthCalibResult result;

  if (inputs.empty()) {
    result.success = true;
    return result;
  }

  // Initialize focal lengths from cameras.
  struct FocalLengthState {
    double optimized = 0.0;
    double initial = 0.0;
  };
  std::unordered_map<camera_t, FocalLengthState> focal_lengths;
  focal_lengths.reserve(cameras.size());
  size_t camera_idx = 0;
  for (const auto& [camera_id, camera] : cameras) {
    ReportProgress(options.progress_callback,
                   "Initializing focal lengths",
                   ++camera_idx,
                   cameras.size());
    const double focal = camera.MeanFocalLength();
    focal_lengths[camera_id] = {focal, focal};
  }

  // Build Ceres problem.
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  auto loss_function = options.CreateLossFunction();

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    ReportProgress(options.progress_callback,
                   "Adding calibration residuals",
                   i + 1,
                   inputs.size());
    if (input.camera_id1 == input.camera_id2) {
      problem.AddResidualBlock(
          FetzerFocalLengthSameCameraCostFunctor::Create(
              input.F, cameras.at(input.camera_id1).PrincipalPoint()),
          loss_function.get(),
          &focal_lengths[input.camera_id1].optimized);
    } else {
      problem.AddResidualBlock(
          FetzerFocalLengthCostFunctor::Create(
              input.F,
              cameras.at(input.camera_id1).PrincipalPoint(),
              cameras.at(input.camera_id2).PrincipalPoint()),
          loss_function.get(),
          &focal_lengths[input.camera_id1].optimized,
          &focal_lengths[input.camera_id2].optimized);
    }
  }

  // Parameterize cameras (fix those with prior, set lower bound).
  size_t num_cameras = 0;
  camera_idx = 0;
  for (const auto& [camera_id, camera] : cameras) {
    ReportProgress(options.progress_callback,
                   "Preparing camera parameters",
                   ++camera_idx,
                   cameras.size());
    double* focal_ptr = &focal_lengths[camera_id].optimized;
    if (!problem.HasParameterBlock(focal_ptr)) continue;

    problem.SetParameterLowerBound(focal_ptr, 0, kFocalLengthLowerBound);
    if (camera.has_prior_focal_length) {
      problem.SetParameterBlockConstant(focal_ptr);
    } else {
      num_cameras++;
    }
  }

  if (num_cameras == 0) {
    VLOG(1) << "No cameras to optimize";
    for (const auto& [camera_id, focal] : focal_lengths) {
      result.focal_lengths[camera_id] = focal.initial;
    }
    result.success = true;
    return result;
  }

  // Set solver options.
  ceres::Solver::Options solver_options = options.solver_options;
  if (cameras.size() < 50) {
    solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  } else {
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  }
  solver_options.num_threads =
      GetEffectiveNumThreads(solver_options.num_threads);
  solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  std::unique_ptr<ViewGraphCalibrationCeresProgressCallback> ceres_progress;
  if (options.progress_callback &&
      !solver_options.minimizer_progress_to_stdout) {
    ceres_progress =
        std::make_unique<ViewGraphCalibrationCeresProgressCallback>(
            options.progress_callback,
            "Solving focal length calibration",
            solver_options.max_num_iterations);
    solver_options.callbacks.push_back(ceres_progress.get());
  }

  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  VLOG(2) << summary.FullReport();

  if (!summary.IsSolutionUsable()) {
    LOG(ERROR) << "Ceres solver failed";
    result.success = false;
    return result;
  }

  // Validate focal lengths and revert degenerate ones.
  size_t rejected_cameras = 0;
  camera_idx = 0;
  for (const auto& [camera_id, camera] : cameras) {
    ReportProgress(options.progress_callback,
                   "Validating focal lengths",
                   ++camera_idx,
                   cameras.size());
    auto& focal = focal_lengths[camera_id];
    if (!problem.HasParameterBlock(&focal.optimized)) continue;

    const double focal_length_ratio = focal.optimized / focal.initial;
    if (focal_length_ratio > options.max_focal_length_ratio ||
        focal_length_ratio < options.min_focal_length_ratio) {
      VLOG(2) << "Ignoring degenerate camera " << camera_id
              << " focal: " << focal.optimized
              << " original focal: " << focal.initial;
      rejected_cameras++;
      // Reset to original focal length.
      focal.optimized = focal.initial;
    }
  }
  VLOG(1) << rejected_cameras << " cameras rejected in view graph calibration";

  for (const auto& [camera_id, focal] : focal_lengths) {
    result.focal_lengths[camera_id] = focal.optimized;
  }

  // Evaluate calibration errors.
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.num_threads = solver_options.num_threads;
  eval_options.apply_loss_function = false;
  std::vector<double> residuals;
  problem.Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);

  size_t residual_idx = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    ReportProgress(options.progress_callback,
                   "Evaluating calibration errors",
                   i + 1,
                   inputs.size());
    const Eigen::Vector2d error(residuals[residual_idx],
                                residuals[residual_idx + 1]);
    result.calibration_errors_sq[input.pair_id] = error.squaredNorm();
    residual_idx += 2;
  }

  result.success = true;
  return result;
}

}  // namespace

std::unique_ptr<ceres::LossFunction>
ViewGraphCalibrationOptions::CreateLossFunction() const {
  return std::make_unique<ceres::CauchyLoss>(loss_function_scale);
}

bool CalibrateViewGraph(const ViewGraphCalibrationOptions& options,
                        Database* database) {
  THROW_CHECK_NOTNULL(database);

  const size_t total_stages =
      6 + (options.cross_validate_prior_focal_lengths ? 1 : 0) +
      (options.reestimate_relative_pose ? 1 : 0);
  ViewGraphCalibrationProgress progress(total_stages);
  ViewGraphCalibrationOptions progress_options = options;
  progress_options.progress_callback = [&progress](const std::string& label,
                                                   const size_t current,
                                                   const size_t total) {
    progress.SetBoundedWork(label, current, total);
  };
  size_t stage_idx = 0;
  auto set_stage = [&progress, &stage_idx](std::string stage) {
    progress.ClearBoundedWork();
    progress.SetStage(++stage_idx, std::move(stage));
  };

  // Read cameras and build image_id -> camera mapping.
  set_stage("Loading database");
  std::unordered_map<camera_t, Camera> cameras;
  auto all_cameras = database->ReadAllCameras();
  for (size_t i = 0; i < all_cameras.size(); ++i) {
    ReportProgress(progress_options.progress_callback,
                   "Loading cameras",
                   i + 1,
                   all_cameras.size());
    Camera& camera = all_cameras[i];
    cameras[camera.camera_id] = std::move(camera);
  }
  std::unordered_map<image_t, const Camera*> image_id_to_camera;
  const auto all_images = database->ReadAllImages();
  for (size_t i = 0; i < all_images.size(); ++i) {
    ReportProgress(progress_options.progress_callback,
                   "Loading images",
                   i + 1,
                   all_images.size());
    const Image& image = all_images[i];
    image_id_to_camera[image.ImageId()] = &cameras.at(image.CameraId());
  }

  // Read UNCALIBRATED and CALIBRATED two-view geometries.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> pairs;
  auto two_view_geometries = database->ReadTwoViewGeometries();
  for (size_t i = 0; i < two_view_geometries.size(); ++i) {
    ReportProgress(progress_options.progress_callback,
                   "Loading two-view geometries",
                   i + 1,
                   two_view_geometries.size());
    auto& [pair_id, tvg] = two_view_geometries[i];
    if (tvg.config == TwoViewGeometry::UNCALIBRATED ||
        tvg.config == TwoViewGeometry::CALIBRATED) {
      pairs.emplace_back(pair_id, std::move(tvg));
    }
  }

  if (pairs.empty()) {
    progress.Finish("No image pairs to calibrate");
    LOG(WARNING) << "No image pairs to calibrate";
    return true;
  }

  if (options.cross_validate_prior_focal_lengths) {
    set_stage("Cross-validating prior focal lengths");
    CrossValidatePriorFocalLengths(options.min_calibrated_pair_ratio,
                                   image_id_to_camera,
                                   pairs,
                                   progress_options.progress_callback);
  }

  // Recompute F from E for CALIBRATED pairs using current calibration.
  set_stage("Recomputing calibrated fundamentals");
  for (size_t i = 0; i < pairs.size(); ++i) {
    auto& [pair_id, tvg] = pairs[i];
    ReportProgress(progress_options.progress_callback,
                   "Recomputing F matrices",
                   i + 1,
                   pairs.size());
    if (tvg.config != TwoViewGeometry::CALIBRATED ||
        !tvg.cam2_from_cam1.has_value()) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const Camera& camera1 = *image_id_to_camera.at(image_id1);
    const Camera& camera2 = *image_id_to_camera.at(image_id2);
    tvg.F = FundamentalFromEssentialMatrix(
        camera2.CalibrationMatrix(),
        EssentialMatrixFromPose(*tvg.cam2_from_cam1),
        camera1.CalibrationMatrix());
  }

  // Prepare inputs and run Ceres optimization.
  set_stage("Preparing focal length calibration");
  std::vector<FocalLengthCalibInput> inputs;
  inputs.reserve(pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto& [pair_id, tvg] = pairs[i];
    ReportProgress(progress_options.progress_callback,
                   "Preparing calibration inputs",
                   i + 1,
                   pairs.size());
    THROW_CHECK(tvg.F.has_value())
        << "Two-view geometry must have F matrix for focal length calibration";
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    inputs.push_back({pair_id,
                      image_id_to_camera.at(image_id1)->camera_id,
                      image_id_to_camera.at(image_id2)->camera_id,
                      tvg.F.value()});
  }

  set_stage("Optimizing focal lengths");
  const FocalLengthCalibResult calib_result =
      CalibrateFocalLengths(progress_options, inputs, cameras);
  if (!calib_result.success) {
    progress.Finish("Failed");
    return false;
  }

  // Update cameras with estimated focal lengths.
  set_stage("Updating cameras");
  size_t camera_idx = 0;
  for (const auto& [camera_id, focal_length] : calib_result.focal_lengths) {
    ReportProgress(progress_options.progress_callback,
                   "Writing cameras",
                   ++camera_idx,
                   calib_result.focal_lengths.size());
    Camera& camera = cameras.at(camera_id);
    camera.SetFocalLength(focal_length);
    camera.has_prior_focal_length = true;
    database->UpdateCamera(camera);
  }

  // Process pairs: tag degenerate or compute E matrix.
  const double max_calibration_error_sq =
      options.max_calibration_error * options.max_calibration_error;
  size_t invalid_counter = 0;
  std::vector<size_t> valid_pair_indices;

  set_stage("Validating two-view geometries");
  for (size_t i = 0; i < pairs.size(); ++i) {
    auto& [pair_id, tvg] = pairs[i];
    ReportProgress(progress_options.progress_callback,
                   "Validating pair calibration",
                   i + 1,
                   pairs.size());
    if (tvg.config != TwoViewGeometry::CALIBRATED &&
        tvg.config != TwoViewGeometry::UNCALIBRATED)
      continue;

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto error_it = calib_result.calibration_errors_sq.find(pair_id);
    if (error_it == calib_result.calibration_errors_sq.end()) {
      continue;
    }

    if (error_it->second > max_calibration_error_sq) {
      invalid_counter++;
      tvg.config = TwoViewGeometry::DEGENERATE;
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    } else {
      THROW_CHECK(tvg.F.has_value())
          << "Two-view geometry must have F matrix for E computation";
      const Camera& camera1 = *image_id_to_camera.at(image_id1);
      const Camera& camera2 = *image_id_to_camera.at(image_id2);
      tvg.E = EssentialFromFundamentalMatrix(camera2.CalibrationMatrix(),
                                             tvg.F.value(),
                                             camera1.CalibrationMatrix());
      tvg.config = TwoViewGeometry::CALIBRATED;
      valid_pair_indices.push_back(i);
    }
  }
  VLOG(1) << "Invalid / total number of two-view geometry: " << invalid_counter
          << " / " << pairs.size();

  // Re-estimate relative poses for valid pairs.
  if (options.reestimate_relative_pose) {
    set_stage("Re-estimating relative poses");
    if (!valid_pair_indices.empty()) {
      std::vector<std::pair<image_pair_t, TwoViewGeometry>> valid_pairs;
      valid_pairs.reserve(valid_pair_indices.size());
      for (size_t i = 0; i < valid_pair_indices.size(); ++i) {
        ReportProgress(progress_options.progress_callback,
                       "Collecting valid pairs",
                       i + 1,
                       valid_pair_indices.size());
        valid_pairs.push_back(std::move(pairs[valid_pair_indices[i]]));
      }
      ReestimateRelativePoses(
          progress_options, valid_pairs, image_id_to_camera, database);
      for (size_t i = 0; i < valid_pairs.size(); ++i) {
        const auto& [pair_id, tvg] = valid_pairs[i];
        ReportProgress(progress_options.progress_callback,
                       "Writing re-estimated pairs",
                       i + 1,
                       valid_pairs.size());
        const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
        database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
      }
    }
  } else {
    for (size_t i = 0; i < valid_pair_indices.size(); ++i) {
      const auto& [pair_id, tvg] = pairs[valid_pair_indices[i]];
      ReportProgress(progress_options.progress_callback,
                     "Writing calibrated pairs",
                     i + 1,
                     valid_pair_indices.size());
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      database->UpdateTwoViewGeometry(image_id1, image_id2, tvg);
    }
  }

  progress.Finish(StringPrintf(
      "Complete (%zu pairs, %zu invalid)", pairs.size(), invalid_counter));
  return true;
}

}  // namespace colmap
