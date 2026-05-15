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

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace colmap {
namespace {

class BundleAdjustmentProgressCallback : public ceres::IterationCallback {
 public:
  BundleAdjustmentProgressCallback(BaseController* controller,
                                   const int max_num_iterations,
                                   const bool render_progress)
      : controller_(controller),
        max_num_iterations_(max_num_iterations),
#ifdef _WIN32
        enabled_(false),
#else
        enabled_(render_progress && max_num_iterations_ > 0 &&
                 isatty(fileno(stderr))),
#endif
        output_fd_(-1) {
    start_time_ = std::chrono::steady_clock::now();
#ifndef _WIN32
    if (enabled_) {
      output_fd_ = dup(fileno(stderr));
    }
#endif
  }

  ~BundleAdjustmentProgressCallback() override {
    std::lock_guard<std::mutex> lock(mutex_);
    ClearLocked();
#ifndef _WIN32
    if (output_fd_ != -1) {
      close(output_fd_);
      output_fd_ = -1;
    }
#endif
  }

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    THROW_CHECK_NOTNULL(controller_);
    if (controller_->CheckIfStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }
    if (IsEnabled()) {
      std::lock_guard<std::mutex> lock(mutex_);
      current_iteration_ = std::max(current_iteration_, summary.iteration);
      RenderLocked();
    }
    return ceres::SOLVER_CONTINUE;
  }

  void Finish(const std::string& status) {
    if (!IsEnabled()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_) {
      return;
    }
    ClearLocked();
    Write(StringPrintf("Bundle adjustment: %s (%d/%d iterations, %.1fs)\n",
                       status.c_str(),
                       std::min(current_iteration_, max_num_iterations_),
                       max_num_iterations_,
                       ElapsedSeconds()));
    finished_ = true;
  }

 private:
  bool IsEnabled() const { return enabled_ && output_fd_ != -1; }

  static std::string MakeBar(const int current,
                             const int total,
                             const int width) {
    const int filled = std::min(width, width * current / total);
    return std::string(filled, '=') + std::string(width - filled, '-');
  }

  double ElapsedSeconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_time_)
        .count();
  }

  void RenderLocked() {
    const int current = std::min(current_iteration_, max_num_iterations_);
    const double percent = 100.0 * current / max_num_iterations_;
    const std::string line =
        FitLine(StringPrintf("Bundle adjustment: %d/%d [%s] %5.1f%% %.1fs",
                             current,
                             max_num_iterations_,
                             MakeBar(current, max_num_iterations_, 20).c_str(),
                             percent,
                             ElapsedSeconds()));
    Write("\r" + line);
    rendered_width_ = std::max(rendered_width_, line.size());
  }

  void ClearLocked() {
    if (!IsEnabled() || rendered_width_ == 0) {
      return;
    }
    Write("\r" + std::string(rendered_width_, ' ') + "\r");
    rendered_width_ = 0;
  }

  void Write(const std::string& text) const {
#ifndef _WIN32
    if (output_fd_ != -1) {
      const ssize_t num_written = write(output_fd_, text.data(), text.size());
      static_cast<void>(num_written);
    }
#endif
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

  BaseController* controller_;
  const int max_num_iterations_;
  const bool enabled_;
  int output_fd_;
  std::chrono::steady_clock::time_point start_time_;
  std::mutex mutex_;
  int current_iteration_ = 0;
  bool finished_ = false;
  size_t rendered_width_ = 0;
};

}  // namespace

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options), reconstruction_(std::move(reconstruction)) {}

void BundleAdjustmentController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);

  LOG_HEADING1("Global bundle adjustment");
  Timer run_timer;
  run_timer.Start();

  if (reconstruction_->NumRegFrames() == 0) {
    LOG(ERROR) << "Need at least one registered frame.";
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  const bool print_summary = ba_options.print_summary;
  ba_options.print_summary = false;

  std::unique_ptr<BundleAdjustmentProgressCallback> progress_callback;
  if (ba_options.ceres) {
    const bool render_progress =
        !VLOG_IS_ON(1) &&
        ba_options.ceres->solver_options.logging_type ==
            ceres::LoggingType::SILENT &&
        !ba_options.ceres->solver_options.minimizer_progress_to_stdout;
    progress_callback = std::make_unique<BundleAdjustmentProgressCallback>(
        this,
        ba_options.ceres->solver_options.max_num_iterations,
        render_progress);
    ba_options.ceres->solver_options.callbacks.push_back(
        progress_callback.get());
  }

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }
  // Fixing the gauge with two cameras leads to a more stable optimization
  // with fewer steps as compared to fixing three points.
  // TODO(jsch): Investigate whether it is safe to not fix the gauge at all,
  // as initial experiments show that it is even faster.
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  // Run bundle adjustment.
  std::unique_ptr<BundleAdjuster> bundle_adjuster =
      CreateDefaultBundleAdjuster(ba_options, ba_config, *reconstruction_);
  const auto summary = bundle_adjuster->Solve();
  if (progress_callback) {
    progress_callback->Finish(summary->num_residuals == 0   ? "skipped"
                              : summary->IsSolutionUsable() ? "complete"
                                                            : "failed");
  }
  if (print_summary || VLOG_IS_ON(1)) {
    if (const auto ceres_summary =
            std::dynamic_pointer_cast<CeresBundleAdjustmentSummary>(summary)) {
      PrintSolverSummary(ceres_summary->ceres_summary,
                         "Bundle adjustment report");
    } else {
      LOG(INFO) << summary->BriefReport();
    }
  }
  reconstruction_->UpdatePoint3DErrors();

  run_timer.PrintMinutes();
}

}  // namespace colmap
