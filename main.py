import turtle
import math

# Function to plot the heart
def plot_heart(a):
    turtle.speed(0)
    turtle.bgcolor("white")
    turtle.pensize(2)
    turtle.color("red")

    turtle.penup()
    for x in range(-100, 100):
        x = x / 100
        y = (x**(2/3) + (math.e/3) * ((math.pi - x**2)**(1/2)) * math.sin(a * math.pi * x))
        turtle.goto(x * 100, y * 100)
        turtle.pendown()

# Main function
def main():
    # Initialize the turtle
    turtle.setup(800, 600)
    turtle.title("Heart Shape using Turtle")
    
    # Parameter 'a' can be adjusted
    a = 1.5
    
    # Plot the heart
    plot_heart(a)
    
    # Hide the turtle and display the window
    turtle.hideturtle()
    turtle.done()

if __name__ == "__main__":
    main()
