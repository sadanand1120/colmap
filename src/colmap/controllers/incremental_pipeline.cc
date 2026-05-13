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

#include "colmap/controllers/incremental_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/database.h"
#include "colmap/util/file.h"
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
#include <unistd.h>
#endif

namespace colmap {

constexpr size_t kMapperLoopNumStages = 6;
constexpr size_t kMapperLoopStageTriangulation = 1;
constexpr size_t kMapperLoopStageLocalRefinement = 2;
constexpr size_t kMapperLoopStageGlobalRefinement = 3;
constexpr size_t kMapperLoopStageColorExtraction = 4;
constexpr size_t kMapperLoopStageSnapshot = 5;
constexpr size_t kMapperLoopStageImageRegistration = 6;

class MapperProgress {
 public:
  explicit MapperProgress(const size_t total_images)
      : total_images_(total_images),
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

  ~MapperProgress() {
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

  void SetStage(std::string stage) {
    std::lock_guard<std::mutex> lock(mutex_);
    stage_ = std::move(stage);
    RenderLocked();
  }

  void SetLoopStage(const size_t stage_index, std::string stage) {
    SetStage(FormatLoopStage(stage_index, std::move(stage)));
  }

  void SetBoundedWork(std::string label,
                      const size_t current,
                      const size_t total,
                      const bool show_single_item = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total == 0 || (total == 1 && !show_single_item)) {
      has_bounded_work_ = false;
      if (current == total) {
        stage_ = std::move(label);
      }
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

  void SetLoopBoundedWork(const size_t stage_index,
                          std::string label,
                          const size_t current,
                          const size_t total,
                          const bool show_single_item = false) {
    if (total == 0 || (total == 1 && !show_single_item)) {
      SetLoopStage(stage_index, std::move(label));
      return;
    }
    SetBoundedWork(std::move(label), current, total, show_single_item);
  }

  void ClearBoundedWork() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_bounded_work_) {
      return;
    }
    has_bounded_work_ = false;
    RenderLocked(true);
  }

  void MarkRegisteredImage(const image_t image_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (image_id != kInvalidImageId &&
        registered_image_ids_.insert(image_id).second) {
      RenderLocked();
    }
  }

  void MarkRegisteredFrame(const Frame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool changed = false;
    for (const data_t& data_id : frame.ImageIds()) {
      if (data_id.id != kInvalidImageId) {
        changed |= registered_image_ids_.insert(data_id.id).second;
      }
    }
    if (changed) {
      RenderLocked();
    }
  }

  void Finish(const std::string& stage) {
    StopHeartbeat();
    size_t num_registered_images = 0;
    std::string elapsed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (finished_) {
        return;
      }
      stage_ = stage;
      has_bounded_work_ = false;
      RenderLocked(true);
      ClearLocked();
      num_registered_images = registered_image_ids_.size();
      elapsed = FormatElapsed();
      finished_ = true;
    }
    if (IsEnabled()) {
      Write(StringPrintf("Mapper: %s (%zu/%zu unique images, %s)\n",
                         stage.c_str(),
                         num_registered_images,
                         total_images_,
                         elapsed.c_str()));
    }
  }

  void ClearForLog() {
    std::lock_guard<std::mutex> lock(mutex_);
    ClearLocked();
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

  static std::string FormatLoopStage(const size_t stage_index,
                                     const std::string& stage) {
    return StringPrintf(
        "(%zu/%zu) %s", stage_index, kMapperLoopNumStages, stage.c_str());
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

    const size_t current = registered_image_ids_.size();
    std::ostringstream line1;
    line1 << "Unique registered images [" << MakeBar(current, total_images_, 28)
          << "] " << current << "/" << total_images_ << " "
          << FormatPercent(current, total_images_) << " | " << stage_ << " | "
          << FormatElapsed();

    std::ostringstream line2;
    if (has_bounded_work_) {
      line2 << "  " << bounded_label_ << " ["
            << MakeBar(bounded_current_, bounded_total_, 24) << "] "
            << bounded_current_ << "/" << bounded_total_ << " "
            << FormatPercent(bounded_current_, bounded_total_);
    }

    ClearLocked();
    Write(line1.str());
    rendered_lines_ = 1;
    if (has_bounded_work_) {
      Write("\n" + line2.str());
      rendered_lines_ = 2;
    }
    last_render_at_ = now;
  }

  static constexpr auto kMinRenderInterval = std::chrono::milliseconds(120);
  static constexpr auto kHeartbeatInterval = std::chrono::seconds(1);

  Timer timer_;
  const size_t total_images_;
  const bool enabled_;
  int output_fd_;
  std::atomic<bool> heartbeat_stop_{false};
  std::thread heartbeat_thread_;
  std::mutex mutex_;
  bool finished_;
  size_t rendered_lines_;
  std::chrono::steady_clock::time_point last_render_at_ =
      std::chrono::steady_clock::time_point::min();
  std::string stage_ = "Starting";
  std::unordered_set<image_t> registered_image_ids_;
  bool has_bounded_work_ = false;
  std::string bounded_label_;
  size_t bounded_current_ = 0;
  size_t bounded_total_ = 0;
};

namespace {

struct MapperBundleAdjustmentProgressState {
  std::mutex mutex;
  int num_solves_started = 0;
};

class MapperBundleAdjustmentProgressCallback : public ceres::IterationCallback {
 public:
  MapperBundleAdjustmentProgressCallback(
      MapperProgress* progress,
      std::shared_ptr<MapperBundleAdjustmentProgressState> state,
      std::string label,
      const int max_num_iterations,
      const int max_num_solves)
      : progress_(progress),
        state_(std::move(state)),
        label_(std::move(label)),
        max_num_iterations_(std::max(max_num_iterations, 1)),
        max_num_solves_(std::max(max_num_solves, 1)) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    solve_index_ = ++state_->num_solves_started;
  }

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    if (progress_ != nullptr) {
      progress_->SetBoundedWork(
          StringPrintf("%s solve %d/%d",
                       label_.c_str(),
                       std::min(solve_index_, max_num_solves_),
                       max_num_solves_),
          std::min(summary.iteration, max_num_iterations_),
          max_num_iterations_);
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  MapperProgress* progress_;
  std::shared_ptr<MapperBundleAdjustmentProgressState> state_;
  const std::string label_;
  const int max_num_iterations_;
  const int max_num_solves_;
  int solve_index_ = 0;
};

void AddMapperBundleAdjustmentProgress(const std::string& label,
                                       const int max_num_solves,
                                       MapperProgress* progress,
                                       BundleAdjustmentOptions* options) {
  if (progress == nullptr || options == nullptr || !options->ceres ||
      options->ceres->solver_options.logging_type !=
          ceres::LoggingType::SILENT ||
      options->ceres->solver_options.minimizer_progress_to_stdout) {
    return;
  }

  auto state = std::make_shared<MapperBundleAdjustmentProgressState>();
  options->ceres->progress_callback_factory =
      [progress, state, label, max_num_solves](
          const ceres::Solver::Options& solver_options) {
        return std::make_unique<MapperBundleAdjustmentProgressCallback>(
            progress,
            state,
            label,
            solver_options.max_num_iterations,
            max_num_solves);
      };
}

void CustomizeIncrementalPipelineOptions(const DatabaseCache& database_cache,
                                         IncrementalPipelineOptions& options) {
  // If the total number of images is small then do not enforce the
  // minimum model size so that we can reconstruct small image
  // collections, i.e., if the model is at least half of the total number
  // of images, we always keep it.
  options.min_model_size = std::min<size_t>(0.5 * database_cache.NumImages(),
                                            options.min_model_size);
}

DatabaseCache::Options CreateDatabaseCacheOptions(
    const IncrementalPipelineOptions& options,
    const ReconstructionManager& reconstruction_manager) {
  DatabaseCache::Options database_cache_options;
  database_cache_options.min_num_matches =
      static_cast<size_t>(options.min_num_matches);
  database_cache_options.ignore_watermarks = options.ignore_watermarks;
  database_cache_options.image_names = {options.image_names.begin(),
                                        options.image_names.end()};
  // Make sure images of the given reconstruction are also included when
  // manually specifying images for the reconstruction procedure.
  if (reconstruction_manager.Size() == 1 && !options.image_names.empty()) {
    const auto& reconstruction = reconstruction_manager.Get(0);
    for (const image_t image_id : reconstruction->RegImageIds()) {
      const auto& image = reconstruction->Image(image_id);
      database_cache_options.image_names.insert(image.Name());
    }
  }
  database_cache_options.load_all_images = options.load_all_images;
  database_cache_options.convert_pose_priors_to_enu =
      options.use_prior_position;
  return database_cache_options;
}

void IterativeGlobalRefinement(MapperProgress* progress,
                               const IncrementalPipelineOptions& options,
                               const IncrementalMapper::Options& mapper_options,
                               IncrementalMapper& mapper,
                               const bool loop_stage = true) {
  if (progress != nullptr) {
    progress->ClearBoundedWork();
    if (loop_stage) {
      progress->SetLoopStage(
          kMapperLoopStageGlobalRefinement,
          "Global refinement: retriangulation and bundle adjustment");
    } else {
      progress->SetStage(
          "Final global refinement: retriangulation and bundle adjustment");
    }
  }
  BundleAdjustmentOptions ba_options = options.GlobalBundleAdjustment();
  const int max_num_ba_solves =
      options.ba_global_max_refinements *
      (mapper_options.ba_global_ignore_redundant_points3D ? 2 : 1);
  AddMapperBundleAdjustmentProgress(
      loop_stage ? "Global BA" : "Final global BA",
      max_num_ba_solves,
      progress,
      &ba_options);
  mapper.IterativeGlobalRefinement(options.ba_global_max_refinements,
                                   options.ba_global_max_refinement_change,
                                   mapper_options,
                                   ba_options,
                                   options.Triangulation());
  if (progress != nullptr) {
    progress->ClearBoundedWork();
    if (loop_stage) {
      progress->SetLoopStage(kMapperLoopStageGlobalRefinement,
                             "Global refinement: filtering frames");
    } else {
      progress->SetStage("Final global refinement: filtering frames");
    }
  }
  mapper.FilterFrames(mapper_options);
  if (progress != nullptr) {
    progress->ClearBoundedWork();
  }
}

void ExtractColors(const std::filesystem::path& image_path,
                   const image_t image_id,
                   Reconstruction& reconstruction,
                   MapperProgress* progress = nullptr) {
  if (!reconstruction.ExtractColorsForImage(image_id, image_path)) {
    if (progress != nullptr) {
      progress->ClearForLog();
    }
    LOG(WARNING) << "Could not read image "
                 << reconstruction.Image(image_id).Name() << " at path "
                 << image_path << ".";
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::filesystem::path& snapshot_path) {
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const auto path = snapshot_path / StringPrintf("%010zu", timestamp);
  CreateDirIfNotExists(path);
  VLOG(1) << "=> Writing to " << path;
  reconstruction.Write(path);
}

bool HasUnknownSensorFromRig(const Reconstruction& reconstruction) {
  std::unordered_set<const Rig*> parameterized_rigs;
  for (const auto& [_, image] : reconstruction.Images()) {
    parameterized_rigs.insert(image.FramePtr()->RigPtr());
  }
  for (const Rig* rig : parameterized_rigs) {
    for (const auto& [sensor_id, sensor_from_rig] : rig->NonRefSensors()) {
      if (sensor_id.type == SensorType::CAMERA &&
          !sensor_from_rig.has_value()) {
        return true;
      }
    }
  }
  return false;
}

size_t CountFrameImages(const Frame& frame) {
  size_t count = 0;
  for (const data_t& data_id : frame.ImageIds()) {
    (void)data_id;
    count += 1;
  }
  return count;
}

}  // namespace

IncrementalMapper::Options IncrementalPipelineOptions::Mapper() const {
  IncrementalMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.fix_existing_frames = fix_existing_frames;
  options.constant_rigs = constant_rigs;
  options.constant_cameras = constant_cameras;
  options.use_prior_position = use_prior_position;
  options.use_robust_loss_on_prior_position = use_robust_loss_on_prior_position;
  options.prior_position_loss_scale = prior_position_loss_scale;
  options.random_seed = random_seed;
  return options;
}

IncrementalTriangulator::Options IncrementalPipelineOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.random_seed = random_seed;
  return options;
}

BundleAdjustmentOptions IncrementalPipelineOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.print_summary = false;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
  if (options.ceres) {
    options.ceres->solver_options.function_tolerance =
        ba_local_function_tolerance;
    options.ceres->solver_options.gradient_tolerance = 10.0;
    options.ceres->solver_options.parameter_tolerance = 0.0;
    options.ceres->solver_options.max_num_iterations =
        ba_local_max_num_iterations;
    options.ceres->solver_options.max_linear_solver_iterations = 100;
    options.ceres->solver_options.logging_type = ceres::LoggingType::SILENT;
    options.ceres->solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
    options.ceres->solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
    options.ceres->min_num_residuals_for_cpu_multi_threading =
        ba_min_num_residuals_for_cpu_multi_threading;
    options.ceres->loss_function_scale = 1.0;
    options.ceres->loss_function_type =
        CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1;
    options.ceres->use_gpu = ba_use_gpu;
    options.ceres->gpu_index = ba_gpu_index;
  }
  return options;
}

BundleAdjustmentOptions IncrementalPipelineOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.print_summary = false;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.refine_sensor_from_rig = ba_refine_sensor_from_rig;
  if (options.ceres) {
    options.ceres->solver_options.function_tolerance =
        ba_global_function_tolerance;
    options.ceres->solver_options.gradient_tolerance = 1.0;
    options.ceres->solver_options.parameter_tolerance = 0.0;
    options.ceres->solver_options.max_num_iterations =
        ba_global_max_num_iterations;
    options.ceres->solver_options.max_linear_solver_iterations = 100;
    options.ceres->solver_options.logging_type = ceres::LoggingType::SILENT;
    if (VLOG_IS_ON(2)) {
      options.ceres->solver_options.minimizer_progress_to_stdout = true;
      options.ceres->solver_options.logging_type =
          ceres::LoggingType::PER_MINIMIZER_ITERATION;
    }
    options.ceres->solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
    options.ceres->solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
    options.ceres->min_num_residuals_for_cpu_multi_threading =
        ba_min_num_residuals_for_cpu_multi_threading;
    options.ceres->loss_function_type =
        CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    options.ceres->use_gpu = ba_use_gpu;
    options.ceres->gpu_index = ba_gpu_index;
  }
  return options;
}

bool IncrementalPipelineOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_frames_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_frames_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GE(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_frames_freq, 0);
  CHECK_OPTION_GT(prior_position_loss_scale, 0.);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_GE(random_seed, -1);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

IncrementalPipeline::IncrementalPipeline(
    std::shared_ptr<IncrementalPipelineOptions> options,
    std::shared_ptr<class Database> database,
    std::shared_ptr<class ReconstructionManager> reconstruction_manager)
    : options_(std::move(THROW_CHECK_NOTNULL(options))),
      reconstruction_manager_(
          THROW_CHECK_NOTNULL(std::move(reconstruction_manager))),
      total_run_timer_(std::make_shared<Timer>()) {
  THROW_CHECK(options_->Check());
  THROW_CHECK_NOTNULL(database);

  LOG(INFO) << "Loading database";
  Timer timer;
  timer.Start();
  database_cache_ = DatabaseCache::Create(
      *database,
      CreateDatabaseCacheOptions(*options_, *reconstruction_manager_));
  timer.PrintMinutes();

  CustomizeIncrementalPipelineOptions(*database_cache_, *options_);
  progress_ = std::make_unique<MapperProgress>(database_cache_->NumImages());

  RegisterCallbacks();
}

IncrementalPipeline::IncrementalPipeline(
    std::shared_ptr<IncrementalPipelineOptions> options,
    std::shared_ptr<class DatabaseCache> database_cache,
    std::shared_ptr<class ReconstructionManager> reconstruction_manager)
    : options_(std::move(THROW_CHECK_NOTNULL(options))),
      reconstruction_manager_(
          THROW_CHECK_NOTNULL(std::move(reconstruction_manager))),
      total_run_timer_(std::make_shared<Timer>()) {
  THROW_CHECK(options_->Check());
  THROW_CHECK_NOTNULL(database_cache);

  database_cache_ = DatabaseCache::CreateFromCache(
      *database_cache,
      CreateDatabaseCacheOptions(*options_, *reconstruction_manager_));

  CustomizeIncrementalPipelineOptions(*database_cache_, *options_);
  progress_ = std::make_unique<MapperProgress>(database_cache_->NumImages());

  RegisterCallbacks();
}

IncrementalPipeline::~IncrementalPipeline() = default;

void IncrementalPipeline::Run() {
  total_run_timer_->Start();
  progress_->SetStage("Starting mapper");

  if (database_cache_->NumImages() == 0) {
    progress_->ClearForLog();
    LOG(WARNING) << "No images with matches";
    return;
  }

  if (options_->use_prior_position && database_cache_->NumPosePriors() == 0) {
    progress_->ClearForLog();
    LOG(WARNING) << "No pose priors";
    return;
  }

  // Is there a sub-reconstruction before we start the reconstruction? I.e. the
  // user has imported an existing reconstruction.
  const bool continue_reconstruction = reconstruction_manager_->Size() > 0;
  THROW_CHECK_LE(reconstruction_manager_->Size(), 1)
      << "Can only continue from a single reconstruction, "
         "but multiple are given.";

  const size_t num_images = database_cache_->NumImages();

  IncrementalMapper::Options mapper_options = options_->Mapper();
  IncrementalMapper mapper(database_cache_);
  if (Reconstruct(mapper,
                  mapper_options,
                  /*continue_reconstruction=*/continue_reconstruction) ==
      Status::STOP) {
    progress_->Finish("Complete");
    total_run_timer_->PrintMinutes();
    return;
  }

  auto ShouldStop = [this, &mapper, &num_images]() {
    return mapper.NumTotalRegImages() == num_images || CheckIfStopped() ||
           CheckReachedMaxRuntime();
  };

  const size_t kNumInitRelaxations = 2;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    if (ShouldStop()) {
      break;
    }

    progress_->SetStage("Relaxing initialization inlier threshold");
    progress_->ClearBoundedWork();
    mapper_options.init_min_num_inliers /= 2;
    mapper.ResetInitializationStats();
    if (Reconstruct(mapper,
                    mapper_options,
                    /*continue_reconstruction=*/false) == Status::STOP) {
      break;
    }

    if (ShouldStop()) {
      break;
    }

    progress_->SetStage("Relaxing initialization triangulation threshold");
    progress_->ClearBoundedWork();
    mapper_options.init_min_tri_angle /= 2;
    mapper.ResetInitializationStats();
    if (Reconstruct(mapper,
                    mapper_options,
                    /*continue_reconstruction=*/false) == Status::STOP) {
      break;
    }
  }

  progress_->Finish("Complete");
  total_run_timer_->PrintMinutes();
}

IncrementalPipeline::Status IncrementalPipeline::InitializeReconstruction(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    Reconstruction& reconstruction) {
  image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
  image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

  // Try to find good initial pair.
  Rigid3d cam2_from_cam1;
  if (!options_->IsInitialPairProvided()) {
    progress_->SetStage("Finding initial image pair");
    progress_->ClearBoundedWork();
    const bool find_init_success = mapper.FindInitialImagePair(
        mapper_options, image_id1, image_id2, cam2_from_cam1);
    if (!find_init_success) {
      progress_->SetStage("No initial image pair found");
      return Status::NO_INITIAL_PAIR;
    }
  } else {
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      progress_->SetStage(StringPrintf(
          "Initial image pair #%d/#%d missing", image_id1, image_id2));
      return Status::NO_INITIAL_PAIR;
    }
    progress_->SetStage(StringPrintf(
        "Checking initial image pair #%d/#%d", image_id1, image_id2));
    progress_->ClearBoundedWork();
    const bool provided_init_success = mapper.EstimateInitialTwoViewGeometry(
        mapper_options, image_id1, image_id2, cam2_from_cam1);
    if (!provided_init_success) {
      progress_->SetStage("Provided initial pair is unsuitable");
      return Status::BAD_INITIAL_PAIR;
    }
  }

  progress_->SetStage(StringPrintf(
      "Registering initial image pair #%d/#%d", image_id1, image_id2));
  progress_->ClearBoundedWork();
  mapper.RegisterInitialImagePair(
      mapper_options, image_id1, image_id2, cam2_from_cam1);
  progress_->MarkRegisteredFrame(*reconstruction.Image(image_id1).FramePtr());
  progress_->MarkRegisteredFrame(*reconstruction.Image(image_id2).FramePtr());

  IncrementalTriangulator::Options tri_options = options_->Triangulation();
  tri_options.min_angle = mapper_options.init_min_tri_angle;
  size_t init_num_views = 0;
  for (const image_t image_id : {image_id1, image_id2}) {
    init_num_views +=
        CountFrameImages(*reconstruction.Image(image_id).FramePtr());
  }
  size_t tri_image_idx = 0;
  for (const image_t image_id : {image_id1, image_id2}) {
    const Image& image = reconstruction.Image(image_id);
    for (const data_t& data_id : image.FramePtr()->ImageIds()) {
      progress_->SetBoundedWork(
          "Initial triangulation", ++tri_image_idx, init_num_views);
      mapper.TriangulateImage(tri_options, data_id.id);
    }
  }
  progress_->ClearBoundedWork();

  if (reconstruction.NumPoints3D() == 0) {
    return Status::BAD_INITIAL_PAIR;
  }

  progress_->SetStage("Initial global bundle adjustment");
  BundleAdjustmentOptions ba_options = options_->GlobalBundleAdjustment();
  AddMapperBundleAdjustmentProgress("Initial global BA",
                                    /*max_num_solves=*/1,
                                    progress_.get(),
                                    &ba_options);
  mapper.AdjustGlobalBundle(mapper_options, ba_options);
  progress_->ClearBoundedWork();
  reconstruction.Normalize();
  mapper.FilterPoints(mapper_options);
  mapper.FilterFrames(mapper_options);

  // Initial image pair failed to register.
  if (reconstruction.NumRegFrames() == 0 || reconstruction.NumPoints3D() == 0) {
    return Status::BAD_INITIAL_PAIR;
  }

  // Number of triangulated points not enough for registering future images.
  if (static_cast<int>(reconstruction.NumPoints3D()) <
      mapper_options.abs_pose_min_num_inliers) {
    return Status::BAD_INITIAL_PAIR;
  }

  if (options_->extract_colors) {
    size_t color_idx = 0;
    for (const image_t image_id : {image_id1, image_id2}) {
      const Image& image = reconstruction.Image(image_id);
      for (const data_t& data_id : image.FramePtr()->ImageIds()) {
        progress_->SetBoundedWork(
            "Initial color extraction", ++color_idx, init_num_views);
        ExtractColors(
            options_->image_path, data_id.id, reconstruction, progress_.get());
      }
    }
    progress_->ClearBoundedWork();
  }
  return Status::SUCCESS;
}

bool IncrementalPipeline::CheckRunGlobalRefinement(
    const Reconstruction& reconstruction,
    const size_t ba_prev_num_reg_frames,
    const size_t ba_prev_num_points) {
  return reconstruction.NumRegFrames() >=
             options_->ba_global_frames_ratio * ba_prev_num_reg_frames ||
         reconstruction.NumRegFrames() >=
             options_->ba_global_frames_freq + ba_prev_num_reg_frames ||
         reconstruction.NumPoints3D() >=
             options_->ba_global_points_ratio * ba_prev_num_points ||
         reconstruction.NumPoints3D() >=
             options_->ba_global_points_freq + ba_prev_num_points;
}

IncrementalPipeline::Status IncrementalPipeline::ReconstructSubModel(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    const std::shared_ptr<Reconstruction>& reconstruction) {
  progress_->SetStage("Starting reconstruction");
  progress_->ClearBoundedWork();
  mapper.BeginReconstruction(reconstruction);
  mapper.Triangulator().SetProgressCallback([this](const std::string& label,
                                                   const size_t current,
                                                   const size_t total) {
    progress_->SetBoundedWork(label, current, total);
  });
  for (const frame_t frame_id : reconstruction->RegFrameIds()) {
    progress_->MarkRegisteredFrame(reconstruction->Frame(frame_id));
  }

  if (HasUnknownSensorFromRig(*reconstruction)) {
    progress_->SetStage("Unknown rig sensor pose");
    return Status::UNKNOWN_SENSOR_FROM_RIG;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Register initial pair
  ////////////////////////////////////////////////////////////////////////////

  if (reconstruction->NumRegFrames() == 0) {
    const Status init_status = IncrementalPipeline::InitializeReconstruction(
        mapper, mapper_options, *reconstruction);
    if (init_status != Status::SUCCESS) {
      return init_status;
    }
  }
  Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

  ////////////////////////////////////////////////////////////////////////////
  // Incremental mapping
  ////////////////////////////////////////////////////////////////////////////

  size_t snapshot_prev_num_reg_frames = reconstruction->NumRegFrames();
  size_t ba_prev_num_reg_frames = reconstruction->NumRegFrames();
  size_t ba_prev_num_points = reconstruction->NumPoints3D();

  std::vector<bool> structure_less_flags;
  if (options_->structure_less_registration_only) {
    structure_less_flags = {true};
  } else {
    if (options_->structure_less_registration_fallback) {
      structure_less_flags = {false, true};
    } else {
      structure_less_flags = {false};
    }
  }

  bool reg_next_success = true;
  bool prev_reg_next_success = true;
  do {
    if (CheckIfStopped() || CheckReachedMaxRuntime()) {
      break;
    }

    prev_reg_next_success = reg_next_success;
    reg_next_success = false;
    image_t next_image_id = kInvalidImageId;

    // Try to register next image. Always prefer structure-based registration
    // first, and if that fails, try (less reliable) structure-less
    // registration.
    for (const bool structure_less : structure_less_flags) {
      progress_->ClearBoundedWork();
      progress_->SetLoopStage(kMapperLoopStageImageRegistration,
                              "Image registration");
      const std::vector<image_t> next_images = mapper.FindNextImages(
          mapper_options, /*structure_less=*/structure_less);
      const std::string candidate_label =
          structure_less ? "Structure-less candidates" : "Image candidates";

      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        next_image_id = next_images[reg_trial];
        std::string candidate_detail = StringPrintf(
            "%s: registering image #%d (%d/%d visible points)",
            candidate_label.c_str(),
            next_image_id,
            mapper.ObservationManager().NumVisiblePoints3D(next_image_id),
            mapper.ObservationManager().NumObservations(next_image_id));

        if (structure_less) {
          candidate_detail = StringPrintf(
              "%s: registering image #%d (%d/%d correspondences)",
              candidate_label.c_str(),
              next_image_id,
              mapper.ObservationManager().NumVisibleCorrespondences(
                  next_image_id),
              mapper.ObservationManager().NumCorrespondences(next_image_id));
        }

        progress_->SetLoopBoundedWork(kMapperLoopStageImageRegistration,
                                      candidate_detail,
                                      reg_trial + 1,
                                      next_images.size(),
                                      /*show_single_item=*/true);

        if (structure_less) {
          reg_next_success = mapper.RegisterNextStructureLessImage(
              mapper_options, next_image_id);
        } else {
          reg_next_success =
              mapper.RegisterNextImage(mapper_options, next_image_id);
        }

        if (reg_next_success) {
          break;
        } else {
          progress_->SetLoopBoundedWork(
              kMapperLoopStageImageRegistration,
              StringPrintf("%s: image #%d failed registration",
                           candidate_label.c_str(),
                           next_image_id),
              reg_trial + 1,
              next_images.size(),
              /*show_single_item=*/true);

          // If initial model fails to continue for some time,
          // abort and try different initial pair.
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              reconstruction->NumRegImages() <
                  static_cast<size_t>(options_->min_model_size)) {
            break;
          }
        }
      }

      if (reg_next_success) {
        break;
      }
    }

    if (reg_next_success) {
      const Image& image = reconstruction->Image(next_image_id);
      progress_->ClearBoundedWork();
      progress_->MarkRegisteredFrame(*image.FramePtr());
      size_t image_data_idx = 0;
      const size_t num_frame_images = CountFrameImages(*image.FramePtr());
      progress_->SetLoopStage(
          kMapperLoopStageTriangulation,
          StringPrintf("Triangulation after image #%d", next_image_id));
      for (const data_t& data_id : image.FramePtr()->ImageIds()) {
        progress_->SetLoopBoundedWork(kMapperLoopStageTriangulation,
                                      "Triangulating registered frame",
                                      ++image_data_idx,
                                      num_frame_images);
        mapper.TriangulateImage(options_->Triangulation(), data_id.id);
      }
      progress_->ClearBoundedWork();
      progress_->SetLoopStage(
          kMapperLoopStageLocalRefinement,
          StringPrintf("Local refinement after image #%d", next_image_id));
      BundleAdjustmentOptions ba_options = options_->LocalBundleAdjustment();
      AddMapperBundleAdjustmentProgress("Local BA",
                                        options_->ba_local_max_refinements,
                                        progress_.get(),
                                        &ba_options);
      mapper.IterativeLocalRefinement(options_->ba_local_max_refinements,
                                      options_->ba_local_max_refinement_change,
                                      mapper_options,
                                      ba_options,
                                      options_->Triangulation(),
                                      next_image_id);
      progress_->ClearBoundedWork();

      if (CheckRunGlobalRefinement(
              *reconstruction, ba_prev_num_reg_frames, ba_prev_num_points)) {
        IterativeGlobalRefinement(
            progress_.get(), *options_, mapper_options, mapper);
        ba_prev_num_points = reconstruction->NumPoints3D();
        ba_prev_num_reg_frames = reconstruction->NumRegFrames();
      }

      if (options_->extract_colors) {
        image_data_idx = 0;
        progress_->ClearBoundedWork();
        progress_->SetLoopStage(
            kMapperLoopStageColorExtraction,
            StringPrintf("Color extraction after image #%d", next_image_id));
        for (const data_t& data_id : image.FramePtr()->ImageIds()) {
          progress_->SetLoopBoundedWork(kMapperLoopStageColorExtraction,
                                        "Extracting colors",
                                        ++image_data_idx,
                                        num_frame_images);
          ExtractColors(options_->image_path,
                        data_id.id,
                        *reconstruction,
                        progress_.get());
        }
        progress_->ClearBoundedWork();
      }

      if (options_->snapshot_frames_freq > 0 &&
          reconstruction->NumRegFrames() >=
              options_->snapshot_frames_freq + snapshot_prev_num_reg_frames) {
        progress_->ClearBoundedWork();
        progress_->SetLoopStage(kMapperLoopStageSnapshot,
                                "Writing reconstruction snapshot");
        snapshot_prev_num_reg_frames = reconstruction->NumRegFrames();
        WriteSnapshot(*reconstruction, options_->snapshot_path);
      }

      Callback(NEXT_IMAGE_REG_CALLBACK);
    }

    const size_t max_model_overlap =
        static_cast<size_t>(options_->max_model_overlap);
    if (mapper.NumSharedRegImages() >= max_model_overlap) {
      break;
    }

    // If no image could be registered, try a single final global iterative
    // bundle adjustment and try again to register one image. If this fails
    // once, then exit the incremental mapping.
    if (!reg_next_success && prev_reg_next_success) {
      IterativeGlobalRefinement(
          progress_.get(), *options_, mapper_options, mapper);
    }
  } while (reg_next_success || prev_reg_next_success);

  if (CheckIfStopped() || CheckReachedMaxRuntime()) {
    return Status::INTERRUPTED;
  }

  // Only run final global BA, if last incremental BA was not global.
  if (reconstruction->NumRegFrames() > 0 &&
      reconstruction->NumRegFrames() != ba_prev_num_reg_frames &&
      reconstruction->NumPoints3D() != ba_prev_num_points) {
    IterativeGlobalRefinement(progress_.get(),
                              *options_,
                              mapper_options,
                              mapper,
                              /*loop_stage=*/false);
  }
  return Status::SUCCESS;
}

IncrementalPipeline::Status IncrementalPipeline::Reconstruct(
    IncrementalMapper& mapper,
    const IncrementalMapper::Options& mapper_options,
    bool continue_reconstruction) {
  for (int num_trials = 0; num_trials < options_->init_num_trials;
       ++num_trials) {
    if (CheckIfStopped() || CheckReachedMaxRuntime()) {
      return Status::STOP;
    }
    progress_->SetBoundedWork(
        "Initialization trials", num_trials + 1, options_->init_num_trials);

    const size_t reconstruction_idx =
        (!continue_reconstruction || num_trials > 0)
            ? reconstruction_manager_->Add()
            : 0;
    std::shared_ptr<Reconstruction> reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    const Status status =
        ReconstructSubModel(mapper, mapper_options, reconstruction);
    switch (status) {
      case Status::INTERRUPTED: {
        reconstruction->UpdatePoint3DErrors();
        progress_->SetStage("Keeping reconstruction after interrupt");
        progress_->ClearBoundedWork();
        mapper.EndReconstruction(/*discard=*/false);
        AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                           reconstruction.get());
        return Status::STOP;
      }

      case Status::UNKNOWN_SENSOR_FROM_RIG: {
        progress_->ClearForLog();
        LOG(ERROR)
            << "Discarding reconstruction due to unknown sensor_from_rig "
               "poses. Either explicitly define the poses by configuring the "
               "rigs or first run reconstruction without configured rigs and "
               "then derive the poses from the initial reconstruction for a "
               "subsequent reconstruction with rig constraints. See "
               "documentation for detailed instructions.";
        mapper.EndReconstruction(/*discard=*/true);
        reconstruction_manager_->Delete(reconstruction_idx);
        // If the reconstruction was discarded due to an unknown sensor from
        // rig, we can stop the outer trial loop, because all trials will fail.
        return Status::STOP;
      }

      case Status::BAD_INITIAL_PAIR: {
        progress_->SetStage("Discarding reconstruction: bad initial pair");
        progress_->ClearBoundedWork();
        mapper.EndReconstruction(/*discard=*/true);
        reconstruction_manager_->Delete(reconstruction_idx);
        // If an initial pair was found but it was bad, we discard and attempt
        // to initialize from any of the remaining pairs in the next trials.
        break;
      }

      case Status::NO_INITIAL_PAIR: {
        progress_->SetStage("Discarding reconstruction: no initial pair");
        progress_->ClearBoundedWork();
        mapper.EndReconstruction(/*discard=*/true);
        reconstruction_manager_->Delete(reconstruction_idx);
        // If no pair could be found, we can exit the trial loop, because
        // the next trials in this loop will not find anything unless the
        // initialization thresholds are relaxed. However, by relaxing the
        // constraints in the outer loop we can succeed.
        return Status::CONTINUE;
      }

      case Status::SUCCESS: {
        // Remember the total number of registered images before potentially
        // discarding it below due to small size, so we can exit out of the main
        // loop, if all images were registered.
        const size_t num_reg_images = reconstruction->NumRegImages();
        const size_t total_num_reg_images = mapper.NumTotalRegImages();

        // Always keep the first reconstruction, independent of size.
        if ((options_->multiple_models && reconstruction_manager_->Size() > 1 &&
             num_reg_images < static_cast<size_t>(options_->min_model_size)) ||
            num_reg_images == 0) {
          progress_->SetStage("Discarding reconstruction: insufficient size");
          progress_->ClearBoundedWork();
          mapper.EndReconstruction(/*discard=*/true);
          reconstruction_manager_->Delete(reconstruction_idx);
        } else {
          reconstruction->UpdatePoint3DErrors();
          progress_->SetStage("Keeping successful reconstruction");
          progress_->ClearBoundedWork();
          mapper.EndReconstruction(/*discard=*/false);
          AlignReconstructionToOrigRigScales(database_cache_->Rigs(),
                                             reconstruction.get());
        }

        Callback(LAST_IMAGE_REG_CALLBACK);

        // Check if we should or can reconstruct another sub-model.
        if (!options_->multiple_models ||
            reconstruction_manager_->Size() >=
                static_cast<size_t>(options_->max_num_models) ||
            total_num_reg_images >= database_cache_->NumImages() - 1) {
          return Status::STOP;
        }

        // In case the reconstruction was successful and there are remaining
        // images, we try to reconstruct another sub-model in the next trial.
        break;
      }

      default:
        LOG(FATAL_THROW) << "Unknown reconstruction status.";
    }
  }

  return Status::CONTINUE;
}

void IncrementalPipeline::TriangulateReconstruction(
    const std::shared_ptr<Reconstruction>& reconstruction) {
  THROW_CHECK_GT(database_cache_->NumImages(), 0)
      << "No images with matches found in the database";
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction);

  LOG(INFO) << "Iterative triangulation";
  size_t image_idx = 0;
  for (const image_t image_id : reconstruction->RegImageIds()) {
    const auto& image = reconstruction->Image(image_id);

    LOG(INFO) << StringPrintf(
        "Triangulating image #%d (%d)", image_id, image_idx++);
    const size_t num_existing_points3D = image.NumPoints3D();
    LOG(INFO) << "=> Image sees " << num_existing_points3D << " / "
              << mapper.ObservationManager().NumObservations(image_id)
              << " points";

    mapper.TriangulateImage(options_->Triangulation(), image_id);
    VLOG(1) << "=> Triangulated "
            << (image.NumPoints3D() - num_existing_points3D) << " points";
  }

  LOG(INFO) << "Retriangulation and Global bundle adjustment";
  mapper.IterativeGlobalRefinement(options_->ba_global_max_refinements,
                                   options_->ba_global_max_refinement_change,
                                   options_->Mapper(),
                                   options_->GlobalBundleAdjustment(),
                                   options_->Triangulation(),
                                   /*normalize_reconstruction=*/false);
  mapper.EndReconstruction(/*discard=*/false);

  reconstruction->UpdatePoint3DErrors();

  LOG(INFO) << "Extracting colors";
  reconstruction->ExtractColorsForAllImages(options_->image_path);
}

void IncrementalPipeline::RegisterCallbacks() {
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

bool IncrementalPipeline::CheckReachedMaxRuntime() const {
  if (options_->max_runtime_seconds > 0 &&
      total_run_timer_->ElapsedSeconds() > options_->max_runtime_seconds) {
    progress_->ClearForLog();
    LOG(INFO) << "Reached maximum runtime of " << options_->max_runtime_seconds
              << " seconds.";
    return true;
  }
  return false;
}

}  // namespace colmap
