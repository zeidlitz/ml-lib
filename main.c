#include <math.h>

#include "base.h"
#include "arena.h"
#include "prng.h"

#include "arena.c"
#include "prng.c"

typedef struct {
  u32 rows, cols;
  f32 *data;
} matrix;

matrix *mat_create(mem_arena *arena, u32 rows, u32 cols);
matrix *mat_load(mem_arena *arena, u32 rows, u32 cols, const char *filename);
b32 mat_copy(matrix *dst, matrix *src);
void mat_clear(matrix *mat);
void mat_fill(matrix *mat, f32 x);
void mat_scale(matrix *mat, f32 scale);
f32 mat_sum(matrix *mat);
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b);
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in);
b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q);
b32 mat_relu_add_grad(matrix *out, const matrix *in);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
b32 mat_cross_entropy_add_grad(matrix *out, const matrix *p, const matrix *q);

void draw_mnist_digit(f32 *data);

int main(void) {
  mem_arena *perm_arena = arena_create(GiB(1), MiB(1));
  matrix *train_images =
      mat_load(perm_arena, 60000, 784, "data/train_images.mat");
  matrix *test_images =
      mat_load(perm_arena, 10000, 784, "data/test_images.mat");
  matrix *train_labels = mat_create(perm_arena, 60000, 10);
  matrix *test_labels = mat_create(perm_arena, 10000, 10);

  {
    matrix *train_labels_file =
        mat_load(perm_arena, 60000, 1, "data/train_labels.mat");
    matrix *test_labels_file =
        mat_load(perm_arena, 10000, 1, "data/test_labels.mat");

    for (u32 i = 0; i < 60000; i++) {
      u32 num = train_labels_file->data[i];
      train_labels->data[i * 10 + num] = 1.0f;
    }

    for (u32 i = 0; i < 10000; i++) {
      u32 num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0f;
    }

    draw_mnist_digit(test_images->data);
    for (u32 i = 0; i < 10; i++) {
      printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");
  }
  arena_destroy(perm_arena);
  return 0;
}

void draw_mnist_digit(f32 *data) {
  for (u32 y = 0; y < 28; y++) {
    for (u32 x = 0; x < 28; x++) {
      f32 num = data[x + y * 28];
      u32 col = 232 + (u32)(num * 23);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

matrix *mat_create(mem_arena *arena, u32 rows, u32 cols) {
  matrix *mat = PUSH_STRUCT(arena, matrix);
  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);
  return mat;
}

matrix *mat_load(mem_arena *arena, u32 rows, u32 cols, const char *filename) {
  matrix *mat = mat_create(arena, rows, cols);
  FILE *f = fopen(filename, "rb");
  fseek(f, 0, SEEK_END);
  u64 size = ftell(f);
  fseek(f, 0, SEEK_SET);
  size = MIN(size, sizeof(f32) * rows * cols);
  fread(mat->data, 1, size, f);
  fclose(f);
  return mat;
}

b32 mat_copy(matrix *dst, matrix *src) {
  if (dst->rows != src->rows || dst->cols != src->cols) {
    return false;
  }

  memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);
  return true;
}
void mat_clear(matrix *mat) {
  memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}
void mat_fill(matrix *mat, f32 x) {
  u64 size = mat->rows * mat->cols;
  for (u64 i = 0; i <= size; i++) {
    mat->data[i] = x;
  }
}

void mat_scale(matrix *mat, f32 scale) {
  u64 size = mat->rows * mat->cols;
  for (u64 i = 0; i <= size; i++) {
    mat->data[i] *= scale;
  }
}

f32 mat_sum(matrix *mat) {
  u64 size = mat->rows * mat->cols;
  f32 sum = 0.0f;
  for (u64 i = 0; i <= size; i++) {
    sum += mat->data[i];
  }
  return sum;
}

b32 mat_add(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }

  if (out->rows != a->rows || out->cols != a->cols) {
    return false;
  }

  u64 size = (u64)out->rows * (u64)out->cols;
  for (u64 i = 0; i <= size; i++) {
    out->data[i] = a->data[i] + a->data[i];
  }

  return true;
}

b32 mat_sub(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }

  if (out->rows != a->rows || out->cols != a->cols) {
    return false;
  }

  u64 size = (u64)out->rows * (u64)out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = a->data[i] - b->data[i];
  }

  return true;
}

void _mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

void _mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 j = 0; j < out->cols; j++) {
      for (u64 k = 0; k < a->cols; k++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

void _mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 k = 0; k < a->rows; k++) {
    for (u64 i = 0; i < out->rows; i++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

void _mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 j = 0; j < out->cols; j++) {
      for (u64 k = 0; k < a->rows; k++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

b32 mat_mul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
            b8 transpose_a, b8 transpose_b) {
  u32 a_rows = transpose_a ? a->cols : a->rows;
  u32 a_cols = transpose_a ? a->rows : a->cols;
  u32 b_rows = transpose_b ? b->cols : b->rows;
  u32 b_cols = transpose_b ? b->rows : b->cols;

  if (a_cols != b_rows) {
    return false;
  }
  if (out->rows != a_rows || out->cols != b_cols) {
    return false;
  }

  if (zero_out) {
    mat_clear(out);
  }

  u32 transpose = (transpose_a << 1) | transpose_b;

  switch (transpose) {
  case 0b00: {
    _mat_mul_nn(out, a, b);
  } break;
  case 0b01: {
    _mat_mul_nt(out, a, b);
  } break;
  case 0b10: {
    _mat_mul_tn(out, a, b);
  } break;
  case 0b11: {
    _mat_mul_tt(out, a, b);
  } break;
  }

  return true;
}

b32 mat_relu(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = MAX(0, in->data[i]);
  }
  return true;
}

b32 mat_softmax(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;

  f32 sum = 0.0f;

  for (u64 i = 0; i < size; i++) {
    out->data[i] = expf(in->data[i]);
    sum += out->data[i];
  }

  mat_scale(out, 1.0f / sum);
  return true;
}

b32 mat_cross_entropy(matrix *out, const matrix *p, const matrix *q) {
  if (p->rows != q->rows || p->cols != q->cols) {
    return false;
  }
  if (out->rows != p->rows || out->cols != p->cols) {
    return false;
  }

  u64 size = (u64)out->rows * (u64)out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i]);
  }

  return true;
}
b32 mat_relu_add_grad(matrix *out, const matrix *in);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out);
b32 mat_cross_entropy_add_grad(matrix *out, const matrix *p, const matrix *q);
