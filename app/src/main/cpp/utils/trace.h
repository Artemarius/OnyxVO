#ifndef ONYX_VO_TRACE_H
#define ONYX_VO_TRACE_H

#include <android/trace.h>

namespace onyx {

// RAII ATrace section — shows up in Perfetto / systrace under the given name.
// Zero overhead when tracing is disabled (ATrace_beginSection checks internally).
// Usage:
//   { ScopedTrace trace("OnyxVO::preprocess"); /* work */ }
class ScopedTrace {
public:
    explicit ScopedTrace(const char* name) { ATrace_beginSection(name); }
    ~ScopedTrace() { ATrace_endSection(); }

    ScopedTrace(const ScopedTrace&) = delete;
    ScopedTrace& operator=(const ScopedTrace&) = delete;
};

} // namespace onyx

#endif // ONYX_VO_TRACE_H
