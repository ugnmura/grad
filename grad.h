/*

# grad.h

Single Header Automatic Differentiation Library written in C.

## Quick Example

### Forward Mode

```c
// grad.c
#define GRAD_IMPLEMENTATION
#include "grad.h"
#include <stdio.h>

int main(void) {
  grad_forward_start_scope();
  grad_forward_t x_grad = grad_forward_init(3);
  grad_forward_t x_pow = grad_forward_mul(&x_grad, &x_grad);
  grad_real_t derivative = x_pow.derivative[x_grad.id];
  printf("%.3f\n", derivative);
}
```

```bash
cc -lm -o grad grad.c
./grad
```

### Reverse Mode

```c
// grad.c
#define GRAD_IMPLEMENTATION
#include "grad.h"
#include <stdio.h>

int main(void) {
  grad_reverse_start_scope();
  grad_reverse_t *x_grad = grad_reverse_init(3);
  grad_reverse_t *x_pow = grad_reverse_mul(x_grad, x_grad);
  grad_reverse_backward(x_pow);
  grad_real_t derivative = x_grad->derivative;
  printf("%.3f\n", derivative);
}
```

```bash
cc -lm -o grad grad.c
./grad
```

## Macro Interface

All these macros are `#define`d by the user before including grad.h

### Flags

Enable or disable certain aspects of grad.h

- `GRAD_IMPLEMENTATION` - Enable definitions of the functions. By default only
declarations are included. See
https://github.com/nothings/stb/blob/f58f558c120e9b32c217290b80bad1a0729fbb2c/docs/stb_howto.txt
  for more info.
- `GRAD_USE_DOUBLE` - use double precision for all computation (default float)

### Redefinable Macros

Redefine default behaviors of grad.h

- `GRAD_FORWARD_TAPE_SIZE` - Maximum number of variables / intermediate values
tracked in forward-mode AD. Must be greater than the number of inputs to the
function. (default 64)
- `GRAD_REVERSE_TAPE_SIZE` - Maximum number of nodes allowed in the reverse-mode
computation graph ("tape"). Must be greather than the total number of operations
performed during forward pass. (default 64)

*/

#ifndef GRAD_H_
#define GRAD_H_

#ifdef GRAD_USE_DOUBLE
typedef double grad_real_t;
#define GRAD_EXP exp
#define GRAD_LOG log
#define GRAD_SIN sin
#define GRAD_COS cos
#define GRAD_SQRT sqrt
#define GRAD_POW pow
#else
typedef float grad_real_t;
#define GRAD_EXP expf
#define GRAD_LOG logf
#define GRAD_SIN sinf
#define GRAD_COS cosf
#define GRAD_SQRT sqrtf
#define GRAD_POW powf
#endif

#include <stdlib.h>

#ifndef GRAD_FORWARD_TAPE_SIZE
#define GRAD_FORWARD_TAPE_SIZE 64
#endif // GRAD_FORWARD_TAPE_SIZE

#ifndef GRAD_REVERSE_TAPE_SIZE
#define GRAD_REVERSE_TAPE_SIZE 64
#endif // GRAD_REVERSE_TAPE_SIZE

typedef struct grad_reverse_t grad_reverse_t;
typedef struct grad_forward_t grad_forward_t;

typedef enum grad_reverse_op_t {
  GRAD_OP_NONE,
  GRAD_OP_ADD,
  GRAD_OP_MUL,
  GRAD_OP_NEG,
  GRAD_OP_INV,
  GRAD_OP_SIN,
  GRAD_OP_COS,
  GRAD_OP_EXP,
  GRAD_OP_LOG,
} grad_reverse_op_t;

struct grad_reverse_t {
  grad_real_t value;
  grad_real_t derivative;

  grad_reverse_op_t operation;
  grad_reverse_t *left;
  grad_reverse_t *right;
};

struct grad_forward_t {
  size_t id;
  grad_real_t value;
  grad_real_t derivative[GRAD_FORWARD_TAPE_SIZE];
};

void grad_forward_start_scope();

grad_forward_t grad_forward_init(grad_real_t value);

grad_forward_t grad_forward_add(const grad_forward_t *left,
                                const grad_forward_t *right);
grad_forward_t grad_forward_add_c(const grad_forward_t *grad,
                                  grad_real_t constant);

grad_forward_t grad_forward_mul(const grad_forward_t *left,
                                const grad_forward_t *right);
grad_forward_t grad_forward_mul_c(const grad_forward_t *grad,
                                  grad_real_t constant);

grad_forward_t grad_forward_inv(const grad_forward_t *grad);
grad_forward_t grad_forward_div(const grad_forward_t *left,
                                const grad_forward_t *right);

grad_forward_t grad_forward_neg(const grad_forward_t *grad);
grad_forward_t grad_forward_sub(const grad_forward_t *left,
                                const grad_forward_t *right);

grad_forward_t grad_forward_exp(const grad_forward_t *grad);
grad_forward_t grad_forward_log(const grad_forward_t *grad);

grad_forward_t grad_forward_sin(const grad_forward_t *grad);
grad_forward_t grad_forward_cos(const grad_forward_t *grad);
grad_forward_t grad_forward_tan(const grad_forward_t *grad);

grad_forward_t grad_forward_sqrt(const grad_forward_t *grad);
grad_forward_t grad_forward_pow(const grad_forward_t *grad, grad_real_t e);

void grad_reverse_start_scope();
grad_reverse_t *grad_reverse_init(grad_real_t value);

grad_reverse_t *grad_reverse_add(grad_reverse_t *left, grad_reverse_t *right);
grad_reverse_t *grad_reverse_sub(grad_reverse_t *left, grad_reverse_t *right);

grad_reverse_t *grad_reverse_mul(grad_reverse_t *left, grad_reverse_t *right);
grad_reverse_t *grad_reverse_div(grad_reverse_t *left, grad_reverse_t *right);

void grad_reverse_backward(grad_reverse_t *grad);

#ifdef GRAD_IMPLEMENTATION

#include <assert.h>
#include <math.h>
#include <string.h>

size_t grad_forward_current_id = 0;

void grad_forward_start_scope() { grad_forward_current_id = 0; }

grad_forward_t grad_forward_init(grad_real_t value) {
  assert(grad_forward_current_id < GRAD_FORWARD_TAPE_SIZE);
  grad_forward_t result = {0};
  result.value = value;
  memset(result.derivative, 0, sizeof(grad_real_t) * GRAD_FORWARD_TAPE_SIZE);
  result.id = grad_forward_current_id;
  result.derivative[grad_forward_current_id] = (grad_real_t)1.0;
  grad_forward_current_id += 1;
  return result;
}

grad_forward_t grad_forward_add(const grad_forward_t *left,
                                const grad_forward_t *right) {
  grad_forward_t result = {0};
  result.value = left->value + right->value;
  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = left->derivative[i] + right->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_add_c(const grad_forward_t *grad,
                                  grad_real_t constant) {
  grad_forward_t result = {0};
  result.value = grad->value + constant;
  memcpy(result.derivative, grad->derivative,
         sizeof(grad_real_t) * GRAD_FORWARD_TAPE_SIZE);
  return result;
}

grad_forward_t grad_forward_mul(const grad_forward_t *left,
                                const grad_forward_t *right) {
  grad_forward_t result = {0};
  result.value = left->value * right->value;

  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] =
        left->derivative[i] * right->value + left->value * right->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_mul_c(const grad_forward_t *grad,
                                  const grad_real_t constant) {
  grad_forward_t result = {0};
  result.value = grad->value * constant;

  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = constant * grad->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_inv(const grad_forward_t *grad) {
  grad_forward_t result = {0};
  result.value = (grad_real_t)1.0f / grad->value;

  grad_real_t inv_sq = (grad_real_t)1.0 / (grad->value * grad->value);
  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = -grad->derivative[i] * inv_sq;
  }
  return result;
}

grad_forward_t grad_forward_div(const grad_forward_t *left,
                                const grad_forward_t *right) {
  grad_forward_t right_inv = grad_forward_inv(right);
  return grad_forward_mul(left, &right_inv);
}

grad_forward_t grad_forward_neg(const grad_forward_t *grad) {
  return grad_forward_mul_c(grad, -1);
}

grad_forward_t grad_forward_sub(const grad_forward_t *left,
                                const grad_forward_t *right) {
  grad_forward_t right_neg = grad_forward_neg(right);
  return grad_forward_add(left, &right_neg);
}

grad_forward_t grad_forward_exp(const grad_forward_t *grad) {
  grad_forward_t result = {0};
  result.value = GRAD_EXP(grad->value);
  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = result.value * grad->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_log(const grad_forward_t *grad) {
  grad_forward_t result = {0};
  result.value = GRAD_LOG(grad->value);
  grad_real_t inv = (grad_real_t)1.0 / grad->value;
  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = inv * grad->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_sin(const grad_forward_t *grad) {
  grad_forward_t result = {0};
  result.value = GRAD_SIN(grad->value);
  grad_real_t val = GRAD_COS(grad->value);
  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = val * grad->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_cos(const grad_forward_t *grad) {
  grad_forward_t result = {0};
  result.value = GRAD_COS(grad->value);
  grad_real_t val = -GRAD_SIN(grad->value);
  for (size_t i = 0; i < grad_forward_current_id; i++) {
    result.derivative[i] = val * grad->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_tan(const grad_forward_t *grad) {
  grad_forward_t s = grad_forward_sin(grad);
  grad_forward_t c = grad_forward_cos(grad);
  return grad_forward_div(&s, &c);
}

grad_forward_t grad_forward_sqrt(const grad_forward_t *grad) {
  grad_forward_t result = {0};
  result.value = GRAD_SQRT(grad->value);
  grad_real_t inv = (grad_real_t)0.5 / result.value;
  for (size_t i = 0; i < grad_forward_current_id; ++i) {
    result.derivative[i] = inv * grad->derivative[i];
  }
  return result;
}

grad_forward_t grad_forward_pow(const grad_forward_t *grad, grad_real_t e) {
  grad_forward_t result = {0};
  result.value = GRAD_POW(grad->value, e);
  grad_real_t val = GRAD_POW(grad->value, e - 1);
  for (size_t i = 0; i < grad_forward_current_id; ++i) {
    result.derivative[i] = e * val * grad->derivative[i];
  }
  return result;
}

grad_reverse_t grad_reverse_tape[GRAD_REVERSE_TAPE_SIZE];
size_t grad_reverse_current_id = 0;

void grad_reverse_start_scope() { grad_reverse_current_id = 0; }

grad_reverse_t *grad_reverse_init(grad_real_t value) {
  assert(grad_reverse_current_id < GRAD_REVERSE_TAPE_SIZE);
  grad_reverse_t *result = &grad_reverse_tape[grad_reverse_current_id];
  grad_reverse_current_id += 1;

  result->value = value;
  result->operation = GRAD_OP_NONE;
  result->derivative = (grad_real_t)0.0;

  return result;
}

grad_reverse_t *grad_reverse_add(grad_reverse_t *left, grad_reverse_t *right) {
  grad_reverse_t *result = grad_reverse_init(left->value + right->value);
  result->operation = GRAD_OP_ADD;
  result->left = left;
  result->right = right;
  return result;
}

grad_reverse_t *grad_reverse_mul(grad_reverse_t *left, grad_reverse_t *right) {
  grad_reverse_t *result = grad_reverse_init(left->value * right->value);
  result->operation = GRAD_OP_MUL;
  result->left = left;
  result->right = right;
  return result;
}

grad_reverse_t *grad_reverse_neg(grad_reverse_t *grad) {
  grad_reverse_t *result = grad_reverse_init(-grad->value);
  result->operation = GRAD_OP_NEG;
  result->left = grad;
  return result;
}

grad_reverse_t *grad_reverse_inv(grad_reverse_t *grad) {
  grad_reverse_t *result = grad_reverse_init(1.0 / grad->value);
  result->operation = GRAD_OP_INV;
  result->left = grad;
  return result;
}

grad_reverse_t *grad_reverse_sub(grad_reverse_t *left, grad_reverse_t *right) {
  grad_reverse_t *neg_right = grad_reverse_neg(right);
  return grad_reverse_add(left, neg_right);
}

grad_reverse_t *grad_reverse_div(grad_reverse_t *left, grad_reverse_t *right) {
  grad_reverse_t *inv_right = grad_reverse_inv(right);
  return grad_reverse_mul(left, inv_right);
}

grad_reverse_t *grad_reverse_sin(grad_reverse_t *grad) {
  grad_reverse_t *result = grad_reverse_init(GRAD_SIN(grad->value));
  result->operation = GRAD_OP_SIN;
  result->left = grad;
  return result;
}

grad_reverse_t *grad_reverse_cos(grad_reverse_t *grad) {
  grad_reverse_t *result = grad_reverse_init(GRAD_COS(grad->value));
  result->operation = GRAD_OP_COS;
  result->left = grad;
  return result;
}

grad_reverse_t *grad_reverse_exp(grad_reverse_t *grad) {
  grad_reverse_t *result = grad_reverse_init(GRAD_EXP(grad->value));
  result->operation = GRAD_OP_EXP;
  result->left = grad;
  return result;
}

grad_reverse_t *grad_reverse_log(grad_reverse_t *grad) {
  grad_reverse_t *result = grad_reverse_init(GRAD_LOG(grad->value));
  result->operation = GRAD_OP_LOG;
  result->left = grad;
  return result;
}

void grad_reverse_backward(grad_reverse_t *output) {
  for (size_t i = 0; i < grad_reverse_current_id; ++i) {
    grad_reverse_tape[i].derivative = (grad_real_t)0.0;
  }

  output->derivative = 1.0;

  for (ssize_t i = (ssize_t)grad_reverse_current_id - 1; i >= 0; --i) {
    grad_reverse_t *grad = &grad_reverse_tape[i];

    switch (grad->operation) {
    case GRAD_OP_ADD: {
      grad->left->derivative += grad->derivative;
      grad->right->derivative += grad->derivative;
      break;
    }
    case GRAD_OP_MUL: {
      grad->left->derivative += grad->right->value * grad->derivative;
      grad->right->derivative += grad->left->value * grad->derivative;
      break;
    }
    case GRAD_OP_NEG: {
      grad->left->derivative += -1.0 * grad->derivative;
      break;
    }
    case GRAD_OP_INV: {
      grad->left->derivative +=
          -grad->derivative / (grad->left->value * grad->left->value);
      break;
    }
    case GRAD_OP_SIN: {
      grad->left->derivative += GRAD_COS(grad->left->value) * grad->derivative;
      break;
    }
    case GRAD_OP_COS: {
      grad->left->derivative += -GRAD_SIN(grad->left->value) * grad->derivative;
      break;
    }
    case GRAD_OP_EXP: {
      grad->left->derivative += GRAD_EXP(grad->left->value) * grad->derivative;
      break;
    }
    case GRAD_OP_LOG: {
      grad->left->derivative += grad->derivative / grad->left->value;
      break;
    }
    default:
      break;
    }
  }
}

#endif // GRAD_IMPLEMENTATION

#endif // GRAD_H_

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2025 Eugene Matsumura
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute
this software, either in source code form or as a compiled binary, for any
purpose, commercial or non-commercial, and by any means. In jurisdictions that
recognize copyright laws, the author or authors of this software dedicate any
and all copyright interest in the software to the public domain. We make this
dedication for the benefit of the public at large and to the detriment of our
heirs and successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this software
under copyright law. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
 */
