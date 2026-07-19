import matplotlib.pyplot as plt
import numpy as np

def plot_sine_wave():
    # Generate x values from -2π to 2π
    x = np.linspace(-2 * np.pi, 2 * np.pi, 400)

    # Calculate corresponding y values (sine of x)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Sine Wave')

    # Add title and labels
    plt.title('Sine Wave from -2π to 2π')
    plt.xlabel('x (radians)')
    plt.ylabel('sin(x)')
    plt.legend()

    # Save the plot as sine_wave.png
    plt.savefig('sine_wave.png')

    # Show the plot
    plt.show()

# Call the function to plot the sine wave
plot_sine_wave()