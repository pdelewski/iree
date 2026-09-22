// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// mlir-obs tracing provider: see mlirobs.h.

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/time.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_APPLE)
#define iree_mlirobs_thread_id() ((uint64_t)pthread_mach_thread_np(pthread_self()))
#elif defined(IREE_PLATFORM_LINUX)
#include <sys/syscall.h>
#include <unistd.h>
#define iree_mlirobs_thread_id() ((uint64_t)syscall(__NR_gettid))
#else
#define iree_mlirobs_thread_id() ((uint64_t)(uintptr_t)pthread_self())
#endif

#if IREE_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// Global state: one output file, one lock around writes, lazy open
//===----------------------------------------------------------------------===//

typedef struct iree_mlirobs_global_t {
  FILE* file;            // NULL until MLIR_OBS_RUN_TRACE names a file
  int state;             // 0 = not checked, 1 = enabled, -1 = disabled
  pthread_mutex_t mutex;
} iree_mlirobs_global_t;

static iree_mlirobs_global_t _g = {NULL, 0, PTHREAD_MUTEX_INITIALIZER};

static int iree_mlirobs_enabled(void) {
  if (IREE_LIKELY(_g.state != 0)) return _g.state > 0;
  pthread_mutex_lock(&_g.mutex);
  if (_g.state == 0) {
    const char* path = getenv("MLIR_OBS_RUN_TRACE");
    if (path && *path) {
      _g.file = fopen(path, "a");
    }
    _g.state = _g.file ? 1 : -1;
  }
  pthread_mutex_unlock(&_g.mutex);
  return _g.state > 0;
}

void iree_tracing_mlirobs_initialize(void) { (void)iree_mlirobs_enabled(); }

void iree_tracing_mlirobs_deinitialize(void) {
  if (_g.state <= 0 || !_g.file) return;
  pthread_mutex_lock(&_g.mutex);
  fflush(_g.file);
  pthread_mutex_unlock(&_g.mutex);
}

//===----------------------------------------------------------------------===//
// Per-thread state: a zone stack and an output buffer
//===----------------------------------------------------------------------===//

#define IREE_MLIROBS_MAX_DEPTH 128
#define IREE_MLIROBS_NAME_CAP 96
#define IREE_MLIROBS_TEXT_CAP 160
#define IREE_MLIROBS_BUFFER_CAP (64 * 1024)

typedef struct iree_mlirobs_zone_t {
  int64_t start_ns;
  int16_t name_length;
  int16_t text_length;
  int has_value;      // appended values, in order: 0, 1 or 2 of them
  int64_t value;
  int64_t value2;
  // The enclosing function when the zone carries its own name (a dynamic
  // name, an external zone): the consumer tells a dispatch zone by it
  // (iree_hal_cmd_execute_dispatch_tiles) without trusting the name.
  const char* fn;
  size_t fn_length;
  char name[IREE_MLIROBS_NAME_CAP];
  char text[IREE_MLIROBS_TEXT_CAP];
} iree_mlirobs_zone_t;

typedef struct iree_mlirobs_thread_t {
  uint64_t id;
  int named;         // the `thread` record was written
  uint32_t depth;    // 0 = no zone open
  size_t used;
  char buffer[IREE_MLIROBS_BUFFER_CAP];
  iree_mlirobs_zone_t stack[IREE_MLIROBS_MAX_DEPTH];
} iree_mlirobs_thread_t;

static _Thread_local iree_mlirobs_thread_t* _t = NULL;

static void iree_mlirobs_flush(iree_mlirobs_thread_t* t) {
  if (!t->used) return;
  pthread_mutex_lock(&_g.mutex);
  fwrite(t->buffer, 1, t->used, _g.file);
  pthread_mutex_unlock(&_g.mutex);
  t->used = 0;
}

static iree_mlirobs_thread_t* iree_mlirobs_thread(void) {
  if (IREE_LIKELY(_t != NULL)) return _t;
  iree_mlirobs_thread_t* t =
      (iree_mlirobs_thread_t*)calloc(1, sizeof(iree_mlirobs_thread_t));
  t->id = iree_mlirobs_thread_id();
  _t = t;
  return t;
}

// Appends a JSON string literal (quoted, escaped) to the thread buffer.
// A record is written whole or not at all: iree_mlirobs_begin_record makes
// room for the largest record before its first byte, and nothing inside a
// record flushes, so two threads' records never interleave in the file.
#define IREE_MLIROBS_RECORD_RESERVE 2048

static void iree_mlirobs_begin_record(iree_mlirobs_thread_t* t) {
  if (t->used + IREE_MLIROBS_RECORD_RESERVE > IREE_MLIROBS_BUFFER_CAP)
    iree_mlirobs_flush(t);
}

static void iree_mlirobs_put_string(iree_mlirobs_thread_t* t, const char* s,
                                    size_t n) {
  char* b = t->buffer;
  size_t u = t->used;
  size_t room = IREE_MLIROBS_BUFFER_CAP - u;
  if (room < 8) return;
  if (n * 2 + 4 > room) n = (room - 4) / 2;
  b[u++] = '"';
  for (size_t i = 0; i < n; ++i) {
    unsigned char ch = (unsigned char)s[i];
    if (ch == '"' || ch == '\\') {
      b[u++] = '\\';
      b[u++] = (char)ch;
    } else if (ch < 0x20) {
      b[u++] = ' ';
    } else {
      b[u++] = (char)ch;
    }
  }
  b[u++] = '"';
  t->used = u;
}

static void iree_mlirobs_printf(iree_mlirobs_thread_t* t, const char* fmt,
                                ...) IREE_PRINTF_ATTRIBUTE(2, 3);
static void iree_mlirobs_printf(iree_mlirobs_thread_t* t, const char* fmt,
                                ...) {
  size_t room = IREE_MLIROBS_BUFFER_CAP - t->used;
  if (room < 2) return;
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(t->buffer + t->used, room, fmt, args);
  va_end(args);
  if (n > 0) t->used += iree_min((size_t)n, room - 1);
}

static void iree_mlirobs_ensure_thread_record(iree_mlirobs_thread_t* t) {
  if (t->named) return;
  t->named = 1;
  iree_mlirobs_begin_record(t);
  iree_mlirobs_printf(t, "{\"kind\":\"thread\",\"thread\":%" PRIu64 ",\"name\":\"\"}\n",
                      t->id);
}

//===----------------------------------------------------------------------===//
// Provider entry points
//===----------------------------------------------------------------------===//

void iree_tracing_set_thread_name(const char* name) {
  if (!iree_mlirobs_enabled()) return;
  iree_mlirobs_thread_t* t = iree_mlirobs_thread();
  t->named = 1;
  iree_mlirobs_begin_record(t);
  iree_mlirobs_printf(t, "{\"kind\":\"thread\",\"thread\":%" PRIu64 ",\"name\":", t->id);
  iree_mlirobs_put_string(t, name, strlen(name));
  iree_mlirobs_printf(t, ",\"time_ns\":%" PRId64 "}\n", iree_platform_time_now());
  iree_mlirobs_flush(t);
}

static iree_zone_id_t iree_mlirobs_zone_begin(const char* name,
                                              size_t name_length,
                                              const char* function_name,
                                              size_t function_name_length) {
  if (!iree_mlirobs_enabled()) return 0;
  iree_mlirobs_thread_t* t = iree_mlirobs_thread();
  if (t->depth + 1 >= IREE_MLIROBS_MAX_DEPTH) return 0;
  iree_zone_id_t zone_id = ++t->depth;
  iree_mlirobs_zone_t* z = &t->stack[zone_id];
  z->fn = NULL;
  z->fn_length = 0;
  if (!name || !name_length) {
    name = function_name;
    name_length = function_name_length;
  } else if (function_name && function_name_length &&
             (function_name_length != name_length ||
              memcmp(function_name, name, name_length) != 0)) {
    z->fn = function_name;
    z->fn_length = function_name_length;
  }
  z->name_length = (int16_t)iree_min(name_length, IREE_MLIROBS_NAME_CAP);
  memcpy(z->name, name, z->name_length);
  z->text_length = 0;
  z->has_value = 0;
  z->value2 = 0;
  z->start_ns = iree_platform_time_now();
  return zone_id;
}

IREE_MUST_USE_RESULT iree_zone_id_t
iree_tracing_zone_begin_impl(const iree_tracing_location_t* src_loc,
                             const char* name, size_t name_length) {
  return iree_mlirobs_zone_begin(
      name ? name : src_loc->name, name_length ? name_length : src_loc->name_length,
      src_loc->function_name, src_loc->function_name_length);
}

IREE_MUST_USE_RESULT iree_zone_id_t iree_tracing_zone_begin_external_impl(
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  (void)file_name;
  (void)file_name_length;
  (void)line;
  return iree_mlirobs_zone_begin(name, name_length, function_name,
                                 function_name_length);
}

void iree_tracing_zone_append_text(iree_zone_id_t zone_id, const char* value,
                                   size_t value_length) {
  if (!zone_id || !_t) return;
  iree_mlirobs_zone_t* z = &_t->stack[zone_id];
  size_t room = IREE_MLIROBS_TEXT_CAP - (size_t)z->text_length;
  size_t n = iree_min(value_length, room);
  memcpy(z->text + z->text_length, value, n);
  z->text_length += (int16_t)n;
}

void iree_tracing_zone_append_value_i64(iree_zone_id_t zone_id, int64_t value) {
  if (!zone_id || !_t) return;
  iree_mlirobs_zone_t* z = &_t->stack[zone_id];
  if (z->has_value == 0) {
    z->value = value;
    z->has_value = 1;
  } else if (z->has_value == 1) {
    z->value2 = value;
    z->has_value = 2;
  }  // a third appended value is dropped: the record keeps two
}

void iree_tracing_zone_end(iree_zone_id_t zone_id) {
  if (!zone_id || !_t) return;
  const int64_t end_ns = iree_platform_time_now();
  iree_mlirobs_thread_t* t = _t;
  iree_mlirobs_zone_t* z = &t->stack[zone_id];
  iree_mlirobs_ensure_thread_record(t);
  iree_mlirobs_begin_record(t);
  iree_mlirobs_printf(t, "{\"kind\":\"slice\",\"thread\":%" PRIu64 ",\"depth\":%u,\"name\":",
                      t->id, zone_id - 1);
  iree_mlirobs_put_string(t, z->name, (size_t)z->name_length);
  if (z->fn) {
    iree_mlirobs_printf(t, ",\"fn\":");
    iree_mlirobs_put_string(t, z->fn, z->fn_length);
  }
  if (z->text_length) {
    iree_mlirobs_printf(t, ",\"text\":");
    iree_mlirobs_put_string(t, z->text, (size_t)z->text_length);
  }
  if (z->has_value >= 1) iree_mlirobs_printf(t, ",\"value\":%" PRId64, z->value);
  if (z->has_value >= 2) iree_mlirobs_printf(t, ",\"value2\":%" PRId64, z->value2);
  iree_mlirobs_printf(t, ",\"time_ns\":%" PRId64 ",\"duration_ns\":%" PRId64 "}\n",
                      z->start_ns, end_ns - z->start_ns);
  if (t->depth > 0) --t->depth;
  // The outermost zone on a thread closed: hand the buffer to the file so a
  // worker that parks for a while does not hold its slices hostage.
  if (t->depth == 0) iree_mlirobs_flush(t);
}

void iree_tracing_plot_value_f64(const char* name, double value) {
  if (!iree_mlirobs_enabled()) return;
  iree_mlirobs_thread_t* t = iree_mlirobs_thread();
  iree_mlirobs_begin_record(t);
  iree_mlirobs_printf(t, "{\"kind\":\"plot\",\"thread\":%" PRIu64 ",\"name\":", t->id);
  iree_mlirobs_put_string(t, name, strlen(name));
  iree_mlirobs_printf(t, ",\"time_ns\":%" PRId64 ",\"value\":%g}\n",
                      iree_platform_time_now(), value);
}

void iree_tracing_frame_mark(const char* name) {
  if (!iree_mlirobs_enabled()) return;
  iree_mlirobs_thread_t* t = iree_mlirobs_thread();
  iree_mlirobs_begin_record(t);
  iree_mlirobs_printf(t, "{\"kind\":\"frame\",\"thread\":%" PRIu64 ",\"name\":", t->id);
  iree_mlirobs_put_string(t, name ? name : "", name ? strlen(name) : 0);
  iree_mlirobs_printf(t, ",\"time_ns\":%" PRId64 "}\n", iree_platform_time_now());
  iree_mlirobs_flush(t);
}

void iree_tracing_message_cstring(const char* value, const char* symbol,
                                  uint32_t color) {
  iree_tracing_message_string_view(value, strlen(value), symbol, color);
}

void iree_tracing_message_string_view(const char* value, size_t value_length,
                                      const char* symbol, uint32_t color) {
  (void)color;
  if (!iree_mlirobs_enabled()) return;
  iree_mlirobs_thread_t* t = iree_mlirobs_thread();
  iree_mlirobs_begin_record(t);
  iree_mlirobs_printf(t, "{\"kind\":\"message\",\"thread\":%" PRIu64 ",\"level\":\"%s\",\"text\":",
                      t->id, symbol ? symbol : "");
  iree_mlirobs_put_string(t, value, value_length);
  iree_mlirobs_printf(t, ",\"time_ns\":%" PRId64 "}\n", iree_platform_time_now());
}

#endif  // IREE_TRACING_FEATURES
