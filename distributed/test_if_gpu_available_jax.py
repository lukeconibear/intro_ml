import jax

if jax.default_backend() == 'gpu':
    print(f"Yes, there are GPUs available.")
else:
    print('No, GPUs are not available.')
