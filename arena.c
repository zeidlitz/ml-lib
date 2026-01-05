#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

typedef uint64_t u64;
typedef int32_t  i32;
typedef int      b32;  // 0 = false, nonzero = true

#define MIN(a,b) ((a)<(b)?(a):(b))
#define ALIGN_UP_POW2(x, align) (((x) + ((align)-1)) & ~((align)-1))
#define ARENA_BASE_POS 0
#define ARENA_ALIGN 16

typedef struct {
    unsigned char* buffer;
    u64 size;
    u64 pos;
} mem_arena;

mem_arena* arena_create_malloc(u64 size) {
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

void* arena_push(mem_arena* arena, u64 sz) {
    u64 aligned_pos = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    u64 new_pos = aligned_pos + sz;

    if (new_pos > arena->size) {
        return NULL; // out of memory
    }

    void* ptr = arena->buffer + aligned_pos;
    arena->pos = new_pos;

    // optionally zero memory
    memset(ptr, 0, sz);

    return ptr;
}

void arena_pop(mem_arena* arena, u64 sz) {
    sz = MIN(sz, arena->pos - ARENA_BASE_POS);
    arena->pos -= sz;
}

void arena_clear(mem_arena* arena) {
    arena->pos = ARENA_BASE_POS;
}

void arena_destroy(mem_arena* arena) {
    if (!arena) return;
    free(arena->buffer);
    free(arena);
}


