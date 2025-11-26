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
