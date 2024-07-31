import numpy as np
import matplotlib.pyplot as plt

## creating the function we want to find the global minimum of
def f(x, y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

## plotting the global minimum as X on the function curve graph

## creates two matrices x and y with the coordinates of the curve
## creates 100 equally spaced points between 0 and 5 using np.linspace
## meshgrid creates a two-dimensional grid with the created matrices: an x and y graph
x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))

## z will be the result of the function f(x, y) created above
z = f(x, y)

## finds the minimum value of the z matrix using argmin and
## then finds the corresponding x coordinates to this minimum value using x.ravel
## the same happens for y_min
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

## sets the size of the plot image
plt.figure(figsize=(8, 6))

## plot legend and color features
plt.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()

## marking the minimum z of the function
plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")

## topography of the contour lines of the graph curve
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

## implementing PSO

## creating birds; particles
birds = 20

## position function for each bird
## starting randomly as it initializes with random numbers
## creates the X matrix of size 2x20 representing the positions of each particle
## creates random numbers between 0 and 1 that will then be multiplied by 5 which is
## the maximum range of the graph
X = np.random.rand(2, birds) * 5

## velocity function of each bird
## starting randomly
## same as above but now the numbers are multiplied by 0.1 to
## reduce their amplitude and represent small speeds, like birds searching for something
V = np.random.rand(2, birds) * 0.1

## adding the birds to the graph curve with position and velocity vectors
plt.quiver(X[0], X[1], V[0], V[1], angles='xy', scale_units='xy', scale=0.3, color='blue', label='Velocity')

## spreading the points on the graph with their labels and markers
plt.scatter(X[0], X[1], marker='o', color='red', label='Particles')
plt.legend()
plt.show()

## implementing gbest as the best position of the swarm
## pbest is the best position found individually

## initially, the best position of each particle is its own initial position
## as they have not explored anything yet
pbest = X

## calculates with the objective function we want to find the minimum
pbest_obj = f(X[0], X[1])

## says that the lowest individual position of all birds is the gbest
gbest = pbest[:, pbest_obj.argmin()]

## minimum value of the function found so far
gbest_obj = pbest_obj.min()

## updating the positions and velocities of the birds

## these are the individual and social hyperparameters of the swarm
## control the influences of pbest and gbest on the updates of velocity and position
c1 = c2 = 0.1

## inertia factor that controls the influence of the previous velocity on the update
w = 0.8

## create two random values between 0 and 1
r = np.random.rand(2)

## velocity update equation for each bird
## typical PSO formula
## uses inertia, pbest, gbest, individual and group intelligence as arguments
## after the inertia factor, updates velocities using current positions and best
## individual positions and the best global position
V = w * V + c1 * r[0] * (pbest - X) + c2 * r[1] * (gbest.reshape(-1, 1) - X)

## updates the position according to the previously calculated velocity
X = X + V

## Calculates the values of the objective function f for the new positions of the particles and stores them in obj.
obj = f(X[0], X[1])

## This line updates the best individual positions based on their new objective function values
## It checks if the obj value is less than or equal to the previous pbest_obj value for each particle and 
## updates the pbest positions to the current positions (X) when necessary.
pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]

## Updates the values of the objective function for the best individual positions
## It replaces the old pbest_obj values with the new maximum values between the
## old values and the new objective function values.
pbest_obj = np.array([pbest_obj, obj]).max(axis=0)

## Calculates the best global position by finding the individual position corresponding to the
## minimum value in the pbest_obj matrix
gbest = pbest[:, pbest_obj.argmin()]

## Calculates the minimum value in the pbest_obj matrix to find the objective function value
## at the best global position
gbest_obj = pbest_obj.min()

for i in range(30):
    ## sets the size of the plot image
    plt.figure(figsize=(8, 6))

    ## plot legend and color features
    plt.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar()

    ## marking the minimum z of the function
    plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")

    ## topography of the contour lines of the graph curve
    contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    ## adding the birds to the graph curve with position and velocity vectors
    plt.quiver(X[0], X[1], V[0], V[1], angles='xy', scale_units='xy', scale=0.3, color='blue', label='Velocity')

    ## spreading the points on the graph with their labels and markers
    plt.scatter(X[0], X[1], marker='o', color='red', label='Particles')
    plt.legend()
    plt.show()

    ## implementing gbest as the best position of the swarm
    ## pbest is the best position found individually

    ## initially, the best position of each particle is its own initial position
    ## as they have not explored anything yet
    pbest = X

    ## calculates with the objective function we want to find the minimum
    pbest_obj = f(X[0], X[1])

    ## says that the lowest individual position of all birds is the gbest
    gbest = pbest[:, pbest_obj.argmin()]

    ## minimum value of the function found so far
    gbest_obj = pbest_obj.min()

    ## updating the positions and velocities of the birds

    ## these are the individual and social hyperparameters of the swarm
    ## control the influences of pbest and gbest on the updates of velocity and position
    c1 = c2 = 0.1

    ## inertia factor that controls the influence of the previous velocity on the update
    w = 0.8

    ## create two random values between 0 and 1
    r = np.random.rand(2)

    ## velocity update equation for each bird
    ## typical PSO formula
    ## uses inertia, pbest, gbest, individual and group intelligence as arguments
    ## after the inertia factor, updates velocities using current positions and best
    ## individual positions and the best global position
    V = w * V + c1 * r[0] * (pbest - X) + c2 * r[1] * (gbest.reshape(-1, 1) - X)

    ## updates the position according to the previously calculated velocity
    X = X + V

    ## Calculates the values of the objective function f for the new positions of the particles and stores them in obj.
    obj = f(X[0], X[1])

    ## This line updates the best individual positions based on their new objective function values
    ## It checks if the obj value is less than or equal to the previous pbest_obj value for each particle and 
    ## updates the pbest positions to the current positions (X) when necessary.
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]

    ## Updates the values of the objective function for the best individual positions
    ## It replaces the old pbest_obj values with the new maximum values between the
    ## old values and the new objective function values.
    pbest_obj = np.array([pbest_obj, obj]).max(axis=0)

    ## Calculates the best global position by finding the individual position corresponding to the
    ## minimum value in the pbest_obj matrix
    gbest = pbest[:, pbest_obj.argmin()]

    ## Calculates the minimum value in the pbest_obj matrix to find the objective function value
    ## at the best global position
    gbest_obj = pbest_obj.min()

print("The minumum value of function is", gbest_obj)

print(f'The location of the minimum value of the function is: x = {gbest[0]}, y = {gbest[1]}')

