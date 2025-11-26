#define GRAD_IMPLEMENTATION
#include "grad.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

int main(void) {
  srand((unsigned)time(NULL));
  grad_real_t x = ((grad_real_t)rand() / RAND_MAX) * 20.0 - 10.0;

  float eps = 1e-8;

  for (size_t i = 0; i < 1000; i++) {
    grad_reverse_start_scope();

    // −x^2 + 6x + 3
    grad_reverse_t *x_grad = grad_reverse_init(x);

    // x^2
    grad_reverse_t *s1 = grad_reverse_mul(x_grad, x_grad);
    // -x^2
    grad_reverse_t *s2 = grad_reverse_neg(s1);
    // 6x
    grad_reverse_t *coe_grad = grad_reverse_init(6);
    grad_reverse_t *s3 = grad_reverse_mul(x_grad, coe_grad);
    // -x^2 + 6x
    grad_reverse_t *s4 = grad_reverse_add(s2, s3);
    // -x^2 + 6x + 3
    grad_reverse_t *c = grad_reverse_init(3);
    grad_reverse_t *f = grad_reverse_add(s4, c);

    grad_reverse_backward(f);

    grad_real_t f_prime = x_grad->derivative;

    grad_real_t change = f->value / f_prime;
    if (fabs(change) < eps) {
      break;
    }

    printf("Step %zu:  %f\n", i, x);

    x -= change;
  }

  printf("Root found: %f\n", x);
}
