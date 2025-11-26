#define GRAD_IMPLEMENTATION
#include "grad.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

int main(void) {
  srand((unsigned)time(NULL));
  grad_real_t x = ((grad_real_t)rand() / RAND_MAX) * 10.0 - 5.0;

  float eps = 1e-8;

  for (size_t i = 0; i < 1000; i++) {
    grad_forward_start_scope();

    // −x^2 + 6x + 3
    grad_forward_t x_grad = grad_forward_init(x);

    // x^2
    grad_forward_t s1 = grad_forward_pow(&x_grad, 2);
    // -x^2
    grad_forward_t s2 = grad_forward_neg(&s1);
    // 6x
    grad_forward_t s3 = grad_forward_mul_c(&x_grad, 6);
    // -x^2 + 6x
    grad_forward_t s4 = grad_forward_add(&s2, &s3);
    // -x^2 + 6x + 3
    grad_forward_t f = grad_forward_add_c(&s4, 3);

    grad_real_t f_prime = f.derivative[x_grad.id];

    grad_real_t change = f.value / f_prime;
    if (fabs(change) < eps) {
      break;
    }

    printf("Step %zu:  %f\n", i, x);

    x -= change;
  }

  printf("Root found: %f\n", x);
}
