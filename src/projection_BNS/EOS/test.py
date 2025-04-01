import jax.numpy as jnp
from inference_utils import GWlikelihood_with_masses

test = GWlikelihood_with_masses("HQC18", "Aplus", "1")
params = {}
params["masses_EOS"] = jnp.array([1, 2, 3])
params["Lambdas_EOS"] = jnp.array([1, 2, 3])

# Run a few tests
value = test.evaluate(params, {})
print(value)

value = test.evaluate(params, {})
print(value)

value = test.evaluate(params, {})
print(value)