from time import perf_counter
import os
begin = perf_counter()
def log(*args, **kwargs):
	print(f"{round(perf_counter() - begin, 2)}s - ", *args, **kwargs)

from ai import *
one_hot_Y = one_hot(Y_train) # precompute the one_hot_Y

log("Initialised!!")

def get_loss(A2, one_hot_Y):
	loss = np.mean((A2 - one_hot_Y) ** 2)
	return loss

# values range from -coeff to +coeff
def mutation(coeff):
	W1 = np.random.rand(10, 784) * 2 - 1
	b1 = np.random.rand(10, 1) * 2 - 1
	W2 = np.random.rand(10, 10) * 2 - 1
	b2 = np.random.rand(10, 1) * 2 - 1
	return W1 * coeff, b1 * coeff, W2 * coeff, b2 * coeff

# Backward propagation too
def backward_prop(X, Y, Z1, A1, Z2, A2, W2):
	"""Calculate backward prop derivatives"""
	m = Y.shape[1]
	dZ2 = 2 * (A2 - Y) / m
	dW2 = dZ2 @ A1.T
	db2 = np.sum(dZ2, axis=1, keepdims=True)

	dA1 = W2.T @ dZ2
	dZ1 = dA1 * (A1 > 0)  # ReLU derivative
	# dZ1 = dA1 * A1 * (1 - A1) # Sigmoid derivative
	dW1 = dZ1 @ X.T
	db1 = np.sum(dZ1, axis=1, keepdims=True)

	return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
	"""Apply derivatives"""
	W1 -= lr * dW1
	b1 -= lr * db1
	W2 -= lr * dW2
	b2 -= lr * db2
	return W1, b1, W2, b2

GENLOG_FILE = "generations.txt" # generation number log
def get_made_model() -> tuple:
	"""Tries to read the starting model from W1.npy, b1.npy, W2.npy, b2.npy.
	Raises RuntimeError if not found."""
	try:
		W1 = np.load("W1.npy")
		b1 = np.load("b1.npy")
		W2 = np.load("W2.npy")
		b2 = np.load("b2.npy")
		model = (W1, b1, W2, b2)

		with open(GENLOG_FILE, 'r') as gen_file:
			log(f"Using preexisting model with {gen_file.read()} generations already ran")
		return model
	except FileNotFoundError:
		raise RuntimeError("No model generated yet! Run evolution.py for ~1000 generations to generate!")

def get_starting_model() -> tuple:
	"""Tries to read the starting model from W1.npy, b1.npy, W2.npy, b2.npy
	Makes a new one if not found."""
	try:
		W1 = np.load("W1.npy")
		b1 = np.load("b1.npy")
		W2 = np.load("W2.npy")
		b2 = np.load("b2.npy")
		model = (W1, b1, W2, b2)

		with open(GENLOG_FILE, 'r') as gen_file:
			log(f"Using preexisting model with {gen_file.read()} generations already ran")
	except FileNotFoundError:
		log("No prior models found! Making a new one to start!")
		model = init_params()
	return model

def run_evolution(model: tuple, gen_n: int, size: int) -> tuple:
	"""Run evolution on a model for a number of generations,
	with a set population size.
	Returns the model.
	A generation is a mutation then a selection."""
	global_best_loss = float('inf')

	models = [model] * size
	for gen in range(1, gen_n + 1):
		log(f"Gen #{gen}/{gen_n}")
		# Apply mutation
		for i in range(size):
			W1e, b1e, W2e, b2e = mutation(0.01)
			W1, b1, W2, b2 = models[i]
			W1, b1, W2, b2 = (W1 + W1e, b1 + b1e, W2 + W2e, b2 + b2e)
			models[i] = (W1, b1, W2, b2)

		# Find best model
		best_model = None
		best_loss = float('inf')
		for model in models:
			W1, b1, W2, b2 = model
			Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
			loss = get_loss(A2, one_hot_Y)
			# dev_predictions = get_predictions(A2)
			# log(round(loss, 2), get_accuracy(dev_predictions, Y_dev))
			if loss < best_loss:
				best_loss = loss
				best_model = (W1, b1, W2, b2)
		log(f"Best loss = {round(best_loss, 6)}")

		# Make sure it actually improved
		if best_loss >= global_best_loss:
			log("Still worse than global best loss or did not improve! Retrying...")
			continue
		else:
			global_best_loss = best_loss

		# Log accuracy
		W1, b1, W2, b2 = best_model
		dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
		log("^ Accuracy=", get_accuracy(dev_predictions, Y_dev))

		# Seed next generation of children
		models = [best_model] * size

	return best_model

def run_backprop(model: tuple, epochs: int, lr: float) -> tuple:
	"""Trains model with backprop"""
	W1, b1, W2, b2 = model
	for epoch in range(1, epochs + 1):
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
		loss = get_loss(A2, one_hot_Y)

		dW1, db1, dW2, db2 = backward_prop(X_train, one_hot_Y, Z1, A1, Z2, A2, W2)
		W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

		log(f"Epoch {epoch}/{epochs} - Loss: {round(loss, 6)}")
		dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
		log("^ Accuracy =", get_accuracy(dev_predictions, Y_dev))

	return W1, b1, W2, b2

def increment_gen_log(number: int) -> None:
	"""We also keep a generations.txt which stores the number of generations run on the currently saved model.
	This function increments by a number and creates a new file if it does not already exist."""
	if not os.path.exists(GENLOG_FILE):
		with open(GENLOG_FILE, 'w') as gen_file:
			gen_file.write(str(number))
		return

	with open(GENLOG_FILE, 'r') as gen_file:
		gen_already = int(gen_file.read())

	with open(GENLOG_FILE, 'w') as gen_file:
		gen_file.write(str(number + gen_already))

def save_model(model: tuple, gen_ran: int) -> None:
	"""Save the weights and biases in the numpy file, and updates the generations."""
	W1, b1, W2, b2 = model
	np.save("W1", W1)
	np.save("b1", b1)
	np.save("W2", W2)
	np.save("b2", b2)
	increment_gen_log(gen_ran)

def main():
	model = get_starting_model()
	approach = input("Enter 1 for backpropagation and 2 for evolution: ")

	if approach == "1":
		epochs = int(input("Enter how many epochs to train: "))
		lr = float(input("Enter learning rate (e.g., 0.01): "))
		model = run_backprop(model, epochs, lr)
	elif approach == "2":
		gen_total = int(input("Enter how many generations do you want to run: "))
		model = run_evolution(model, gen_total, 50)
	else:
		raise ValueError("Enter valid approach number!")

	save_model(model, epochs)

if __name__ == "__main__":
	main()