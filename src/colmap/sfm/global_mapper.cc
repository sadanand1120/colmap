#include "colmap/sfm/global_mapper.h"

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/math/union_find.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
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

#ifndef _WIN32
#include <unistd.h>
#endif

namespace colmap {
namespace {

class GlobalMapperProgress {
 public:
  explicit GlobalMapperProgress(const size_t total_stages)
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

  ~GlobalMapperProgress() {
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
    RenderLocked(true);
  }

  void SetBoundedWork(std::string label,
                      const size_t current,
                      const size_t total) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total <= 1) {
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
      Write(StringPrintf("Global mapper: %s (%zu/%zu stages, %s)\n",
                         stage.c_str(),
                         total_stages_,
                         total_stages_,
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
    line1 << "Global mapper [" << MakeBar(stage_index_, total_stages_, 28)
          << "] " << stage_index_ << "/" << total_stages_ << " "
          << FormatPercent(stage_index_, total_stages_) << " | " << stage_
          << " | " << FormatElapsed();

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

struct GlobalMapperCeresProgressState {
  std::mutex mutex;
  int num_solves_started = 0;
};

class GlobalMapperCeresProgressCallback : public ceres::IterationCallback {
 public:
  GlobalMapperCeresProgressCallback(
      GlobalMapper::ProgressCallback progress,
      std::shared_ptr<GlobalMapperCeresProgressState> state,
      std::string label,
      const int max_num_iterations,
      const int max_num_solves)
      : progress_(std::move(progress)),
        state_(std::move(state)),
        label_(std::move(label)),
        max_num_iterations_(std::max(max_num_iterations, 1)),
        max_num_solves_(std::max(max_num_solves, 1)) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    solve_index_ = ++state_->num_solves_started;
  }

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    if (progress_) {
      progress_(StringPrintf("%s solve %d/%d",
                             label_.c_str(),
                             std::min(solve_index_, max_num_solves_),
                             max_num_solves_),
                std::min(summary.iteration, max_num_iterations_),
                max_num_iterations_);
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  GlobalMapper::ProgressCallback progress_;
  std::shared_ptr<GlobalMapperCeresProgressState> state_;
  const std::string label_;
  const int max_num_iterations_;
  const int max_num_solves_;
  int solve_index_ = 0;
};

void AddCeresProgress(const std::string& label,
                      const GlobalMapper::ProgressCallback& progress,
                      BundleAdjustmentOptions* options,
                      const int max_num_solves = 1) {
  if (!progress || options == nullptr || !options->ceres ||
      options->ceres->solver_options.minimizer_progress_to_stdout) {
    return;
  }
  auto state = std::make_shared<GlobalMapperCeresProgressState>();
  options->ceres->progress_callback_factory =
      [progress, state, label, max_num_solves](
          const ceres::Solver::Options& solver_options) {
        return std::make_unique<GlobalMapperCeresProgressCallback>(
            progress,
            state,
            label,
            solver_options.max_num_iterations,
            max_num_solves);
      };
}

void AddCeresProgress(const std::string& label,
                      const GlobalMapper::ProgressCallback& progress,
                      GlobalPositionerOptions* options) {
  if (!progress || options == nullptr ||
      options->solver_options.minimizer_progress_to_stdout) {
    return;
  }
  auto state = std::make_shared<GlobalMapperCeresProgressState>();
  options->progress_callback_factory =
      [progress, state, label](const ceres::Solver::Options& solver_options) {
        return std::make_unique<GlobalMapperCeresProgressCallback>(
            progress,
            state,
            label,
            solver_options.max_num_iterations,
            /*max_num_solves=*/1);
      };
}

void ReportProgress(const GlobalMapper::ProgressCallback& progress,
                    const std::string& label,
                    const size_t current,
                    const size_t total) {
  if (progress && total > 1) {
    progress(label, current, total);
  }
}

size_t CountValidEdges(const PoseGraph& pose_graph) {
  size_t num_valid_edges = 0;
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    (void)pair_id;
    (void)edge;
    ++num_valid_edges;
  }
  return num_valid_edges;
}

bool RunBundleAdjustment(const BundleAdjustmentOptions& options,
                         Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);

  return ba->Solve()->IsSolutionUsable();
}

GlobalMapperOptions InitializeOptions(const GlobalMapperOptions& options) {
  // Propagate random seed and num_threads to component options.
  GlobalMapperOptions opts = options;
  if (opts.random_seed >= 0) {
    opts.rotation_averaging.random_seed = opts.random_seed;
    opts.global_positioning.random_seed = opts.random_seed;
    opts.global_positioning.use_parameter_block_ordering = false;
    opts.retriangulation.random_seed = opts.random_seed;
  }
  opts.global_positioning.solver_options.num_threads = opts.num_threads;
  if (opts.bundle_adjustment.ceres) {
    opts.bundle_adjustment.ceres->solver_options.num_threads = opts.num_threads;
#if CERES_VERSION_MAJOR < 2
    opts.bundle_adjustment.ceres->solver_options.num_linear_solver_threads =
        opts.num_threads;
#endif  // CERES_VERSION_MAJOR
  }
  return opts;
}

}  // namespace

GlobalMapper::GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Read pose priors from the database cache.
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const GlobalMapperOptions& options,
                                   ProgressCallback progress_callback) {
  using Observation = std::pair<image_t, point2D_t>;
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  // Build keypoints map from registered images.
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  const std::vector<image_t> reg_image_ids = reconstruction_->RegImageIds();
  size_t image_idx = 0;
  for (const auto image_id : reg_image_ids) {
    ReportProgress(progress_callback,
                   "Collecting keypoints",
                   ++image_idx,
                   reg_image_ids.size());
    const auto& image = reconstruction_->Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const auto& point2D : image.Points2D()) {
      points.push_back(point2D.xy);
    }
    image_id_to_keypoints.emplace(image_id, std::move(points));
  }

  auto corr_graph = database_cache_->CorrespondenceGraph();

  // Union all matching observations.
  UnionFind<Observation> uf;
  FeatureMatches matches;
  const size_t num_valid_edges = CountValidEdges(*pose_graph_);
  size_t edge_idx = 0;
  for (const auto& [pair_id, edge] : pose_graph_->ValidEdges()) {
    ReportProgress(progress_callback,
                   "Merging matched observations",
                   ++edge_idx,
                   num_valid_edges);
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    corr_graph->ExtractMatchesBetweenImages(image_id1, image_id2, matches);
    for (const auto& match : matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root.
  uf.Compress();
  std::unordered_map<Observation, std::vector<Observation>> track_map;
  for (const auto& [obs, root] : uf.Parents()) {
    track_map[root].push_back(obs);
  }
  VLOG(1) << "Established " << track_map.size() << " tracks from "
          << uf.Parents().size() << " observations";

  // Validate tracks, check consistency, and collect valid ones with lengths.
  std::unordered_map<point3D_t, Point3D> candidate_points3D;
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;

  size_t track_idx = 0;
  for (const auto& [track_id, observations] : track_map) {
    ReportProgress(
        progress_callback, "Validating tracks", ++track_idx, track_map.size());
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy =
          image_id_to_keypoints.at(image_id).at(feature_id);

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          const double sq_threshold =
              options.track_intra_image_consistency_threshold *
              options.track_intra_image_consistency_threshold;
          if ((existing_xy - xy).squaredNorm() > sq_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          ++discarded_counter;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }
      point3D.track.AddElement(image_id, feature_id);
    }

    if (!is_consistent) continue;

    const size_t num_images = image_id_set.size();
    if (num_images < static_cast<size_t>(options.track_min_num_views_per_track))
      continue;

    const point3D_t point3D_id = next_point3D_id++;
    track_lengths.emplace_back(point3D.track.Length(), point3D_id);
    candidate_points3D.emplace(point3D_id, std::move(point3D));
  }

  VLOG(1) << "Kept " << candidate_points3D.size() << " tracks, discarded "
          << discarded_counter << " due to inconsistency";

  // Sort tracks by length (descending) and select for problem.
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  std::unordered_map<image_t, size_t> tracks_per_image;
  size_t images_left = image_id_to_keypoints.size();
  size_t selected_track_idx = 0;
  for (const auto& [track_length, point3D_id] : track_lengths) {
    ReportProgress(progress_callback,
                   "Selecting tracks",
                   ++selected_track_idx,
                   track_lengths.size());
    auto& point3D = candidate_points3D.at(point3D_id);

    // Check if any image in this track still needs more observations.
    const bool should_add = std::any_of(
        point3D.track.Elements().begin(),
        point3D.track.Elements().end(),
        [&](const auto& obs) {
          return tracks_per_image[obs.image_id] <=
                 static_cast<size_t>(options.track_required_tracks_per_view);
        });
    if (!should_add) continue;

    // Update image counts.
    for (const auto& obs : point3D.track.Elements()) {
      auto& count = tracks_per_image[obs.image_id];
      if (count == static_cast<size_t>(options.track_required_tracks_per_view))
        --images_left;
      ++count;
    }

    // Add track after updating counts so we can move.
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));

    if (images_left == 0) break;
  }

  VLOG(1) << "Before filtering: " << candidate_points3D.size()
          << ", after filtering: " << reconstruction_->NumPoints3D();
}

bool GlobalMapper::GlobalPositioning(const GlobalPositionerOptions& options,
                                     double max_angular_reproj_error_deg,
                                     double max_normalized_reproj_error,
                                     double min_tri_angle_deg,
                                     ProgressCallback progress_callback) {
  GlobalPositionerOptions positioning_options = options;
  positioning_options.progress_callback = progress_callback;
  AddCeresProgress(
      "Global positioning", progress_callback, &positioning_options);
  if (!RunGlobalPositioning(
          positioning_options, *pose_graph_, *reconstruction_)) {
    return false;
  }

  // Filter tracks based on the estimation
  ObservationManager obs_manager(*reconstruction_);
  obs_manager.SetProgressCallback(progress_callback);

  // First pass: use relaxed threshold (2x) for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * max_angular_reproj_error_deg,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::ANGULAR);

  // Second pass: apply strict threshold for cameras with prior focal length.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  std::vector<std::pair<image_t, point2D_t>> obs_to_delete;
  const std::unordered_set<point3D_t> point3D_ids =
      reconstruction_->Point3DIds();
  size_t point3D_idx = 0;
  for (const auto point3D_id : point3D_ids) {
    ReportProgress(progress_callback,
                   "Filtering calibrated camera observations",
                   ++point3D_idx,
                   point3D_ids.size());
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const auto& point3D = reconstruction_->Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& image = reconstruction_->Image(track_el.image_id);
      const auto& camera = *image.CameraPtr();
      if (!camera.has_prior_focal_length) {
        continue;
      }
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const double error = CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (error > max_angular_error_rad) {
        obs_to_delete.emplace_back(track_el.image_id, track_el.point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : obs_to_delete) {
    if (reconstruction_->Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
      obs_manager.DeleteObservation(image_id, point2D_idx);
    }
  }

  // Filter tracks based on triangulation angle and reprojection error
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  // Normalize the structure for numerical stability.
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjustmentOptions& options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage,
    ProgressCallback progress_callback) {
  for (int ite = 0; ite < num_iterations; ite++) {
    // Optional fixed-rotation stage: optimize positions only
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      AddCeresProgress(
          StringPrintf(
              "Fixed-rotation BA iteration %d/%d", ite + 1, num_iterations),
          progress_callback,
          &opts_position_only);
      if (!RunBundleAdjustment(opts_position_only, *reconstruction_)) {
        return false;
      }
      VLOG(1) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << ", fixed-rotation stage finished";
    }

    // Joint optimization stage: default BA
    if (!skip_joint_optimization_stage) {
      BundleAdjustmentOptions opts_joint = options;
      AddCeresProgress(
          StringPrintf("Joint BA iteration %d/%d", ite + 1, num_iterations),
          progress_callback,
          &opts_joint);
      if (!RunBundleAdjustment(opts_joint, *reconstruction_)) {
        return false;
      }
    }
    VLOG(1) << "Global bundle adjustment iteration " << ite + 1 << " / "
            << num_iterations << " finished";

    // Normalize the structure for numerical stability.
    // TODO: Skip normalization when position priors are used (similar to
    // incremental mapper's !use_prior_position condition).
    reconstruction_->Normalize();

    // Filter tracks based on the estimation
    // For the filtering, in each round, the criteria for outlier is
    // tightened. If only few tracks are changed, no need to start bundle
    // adjustment right away. Instead, use a more strict criteria to filter
    VLOG(1) << "Filtering tracks by reprojection ...";

    ObservationManager obs_manager(*reconstruction_);
    obs_manager.SetProgressCallback(progress_callback);
    bool status = true;
    size_t filtered_num = 0;
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_normalized_reproj_error,
          reconstruction_->Point3DIds(),
          ReprojectionErrorType::NORMALIZED);

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      VLOG(1) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  // Filter tracks based on the estimation
  VLOG(1) << "Filtering tracks by reprojection ...";
  {
    ObservationManager obs_manager(*reconstruction_);
    obs_manager.SetProgressCallback(progress_callback);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_normalized_reproj_error,
        reconstruction_->Point3DIds(),
        ReprojectionErrorType::NORMALIZED);
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        min_tri_angle_deg, reconstruction_->Point3DIds());
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const IncrementalTriangulator::Options& options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    ProgressCallback progress_callback) {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);
  mapper.Triangulator().SetProgressCallback(progress_callback);

  // Triangulate all registered images.
  const std::vector<image_t> reg_image_ids = reconstruction_->RegImageIds();
  size_t image_idx = 0;
  for (const auto image_id : reg_image_ids) {
    ReportProgress(progress_callback,
                   "Triangulating registered images",
                   ++image_idx,
                   reg_image_ids.size());
    mapper.TriangulateImage(options, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  BundleAdjustmentOptions custom_ba_options = ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_num_iterations = 50;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
    AddCeresProgress("Retriangulation refinement BA",
                     progress_callback,
                     &custom_ba_options,
                     /*max_num_solves=*/5);
  }

  // Iterative global refinement.
  IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   custom_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/true);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  ObservationManager obs_manager(*reconstruction_);
  obs_manager.SetProgressCallback(progress_callback);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  BundleAdjustmentOptions final_ba_options = ba_options;
  AddCeresProgress("Final global BA", progress_callback, &final_ba_options);
  if (!RunBundleAdjustment(final_ba_options, *reconstruction_)) {
    return false;
  }

  // Normalize the structure for numerical stability.
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());

  return true;
}

bool GlobalMapper::Solve(const GlobalMapperOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Propagate random seed and num_threads to component options.
  GlobalMapperOptions opts = InitializeOptions(options);
  const size_t num_stages =
      static_cast<size_t>(!opts.skip_rotation_averaging) +
      static_cast<size_t>(!opts.skip_track_establishment) +
      static_cast<size_t>(!opts.skip_global_positioning) +
      static_cast<size_t>(!opts.skip_bundle_adjustment) +
      static_cast<size_t>(!opts.skip_retriangulation);
  GlobalMapperProgress progress(num_stages);
  auto progress_callback = [&progress](const std::string& label,
                                       const size_t current,
                                       const size_t total) {
    progress.SetBoundedWork(label, current, total);
  };
  size_t stage_idx = 0;

  // Run rotation averaging
  if (!opts.skip_rotation_averaging) {
    progress.ClearForLog();
    LOG_HEADING1("Running rotation averaging");
    progress.SetStage(++stage_idx, "Rotation averaging");
    opts.rotation_averaging.progress_callback = progress_callback;
    Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(opts.rotation_averaging)) {
      progress.Finish("Failed");
      return false;
    }
    progress.ClearBoundedWork();
    progress.ClearForLog();
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!opts.skip_track_establishment) {
    progress.ClearForLog();
    LOG_HEADING1("Running track establishment");
    progress.SetStage(++stage_idx, "Track establishment");
    Timer run_timer;
    run_timer.Start();
    EstablishTracks(opts, progress_callback);
    progress.ClearBoundedWork();
    progress.ClearForLog();
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (!opts.skip_global_positioning) {
    progress.ClearForLog();
    LOG_HEADING1("Running global positioning");
    progress.SetStage(++stage_idx, "Global positioning");
    Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(opts.global_positioning,
                           opts.max_angular_reproj_error_deg,
                           opts.max_normalized_reproj_error,
                           opts.min_tri_angle_deg,
                           progress_callback)) {
      progress.Finish("Failed");
      return false;
    }
    progress.ClearBoundedWork();
    progress.ClearForLog();
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Bundle adjustment
  if (!opts.skip_bundle_adjustment) {
    progress.ClearForLog();
    LOG_HEADING1("Running iterative bundle adjustment");
    progress.SetStage(++stage_idx, "Iterative bundle adjustment");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(opts.bundle_adjustment,
                                   opts.max_normalized_reproj_error,
                                   opts.min_tri_angle_deg,
                                   opts.ba_num_iterations,
                                   opts.ba_skip_fixed_rotation_stage,
                                   opts.ba_skip_joint_optimization_stage,
                                   progress_callback)) {
      progress.Finish("Failed");
      return false;
    }
    progress.ClearBoundedWork();
    progress.ClearForLog();
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!opts.skip_retriangulation) {
    progress.ClearForLog();
    LOG_HEADING1("Running iterative retriangulation and refinement");
    progress.SetStage(++stage_idx, "Iterative retriangulation and refinement");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(opts.retriangulation,
                                         opts.bundle_adjustment,
                                         opts.max_normalized_reproj_error,
                                         opts.min_tri_angle_deg,
                                         progress_callback)) {
      progress.Finish("Failed");
      return false;
    }
    progress.ClearBoundedWork();
    progress.ClearForLog();
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  progress.Finish("Complete");
  return true;
}

}  // namespace colmap
