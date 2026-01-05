#ifndef ARENA_H
#define ARENA_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef uint64_t u64;
typedef int32_t  i32;
typedef int      b32;  // 0 = false, nonzero = true

#define ARENA_BASE_POS 0
#define ARENA_ALIGN 16

typedef struct {
    unsigned char* buffer;
    u64 size;
    u64 pos;
} mem_arena;

typedef struct {
    mem_arena* arena;
    u64 start_pos;
} mem_arena_temp;

static inline mem_arena* arena_create(u64 size) {
    mem_arena* arena = malloc(sizeof(mem_arena));
    if (!arena) return NULL;

    arena->buffer = malloc(size);
    if (!arena->buffer) {
        free(arena);
        return NULL;
    }

    arena->size = size;
    arena->pos = 0;
    return arena;
}

static inline void arena_destroy(mem_arena* arena) {
    if (!arena) return;
    free(arena->buffer);
    free(arena);
}

static inline void* arena_push(mem_arena* arena, u64 sz, b32 non_zero) {
    u64 aligned_pos = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    u64 new_pos = aligned_pos + sz;

    if (new_pos > arena->size) return NULL;

    void* ptr = arena->buffer + aligned_pos;
    arena->pos = new_pos;

    if (!non_zero) memset(ptr, 0, sz);

    return ptr;
}

static inline void arena_pop(mem_arena* arena, u64 sz) {
    sz = MIN(sz, arena->pos - ARENA_BASE_POS);
    arena->pos -= sz;
}

static inline void arena_pop_to(mem_arena* arena, u64 pos) {
    u64 sz = pos < arena->pos ? arena->pos - pos : 0;
    arena_pop(arena, sz);
}

static inline void arena_clear(mem_arena* arena) {
    arena->pos = ARENA_BASE_POS;
}

static inline mem_arena_temp arena_temp_begin(mem_arena* arena) {
    mem_arena_temp temp = { .arena = arena, .start_pos = arena->pos };
    return temp;
}

static inline void arena_temp_end(mem_arena_temp temp) {
    arena_pop_to(temp.arena, temp.start_pos);
}

static inline mem_arena_temp arena_scratch_get(mem_arena** conflicts, u32 num_conflicts) {
    static __thread mem_arena* _scratch_arenas[2] = {NULL, NULL};
    i32 scratch_index = -1;

    for (i32 i = 0; i < 2; i++) {
        b32 conflict_found = 0;
        for (u32 j = 0; j < num_conflicts; j++) {
            if (_scratch_arenas[i] == conflicts[j]) {
                conflict_found = 1;
                break;
            }
        }
        if (!conflict_found) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) return (mem_arena_temp){0};

    mem_arena** selected = &_scratch_arenas[scratch_index];
    if (*selected == NULL) {
        *selected = arena_create(MiB(64));
    }

    return arena_temp_begin(*selected);
}

static inline void arena_scratch_release(mem_arena_temp scratch) {
    arena_temp_end(scratch);
}

#endif
