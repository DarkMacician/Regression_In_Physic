from manim import *

class BilliardSimulation(Scene):
    def construct(self):
        # Create the billiard table (rectangle)
        table = Rectangle(width=6, height=3, color=WHITE)
        table.set_fill(BLUE, opacity=0.5)

        # Create a ball (circle)
        ball = Circle(radius=0.25, color=YELLOW)
        ball.move_to(LEFT * 2 + UP * 1)  # Place the ball at a starting position

        # Set the ball's initial velocity (simple vector)
        velocity = [0.5, 0.3]  # Moving towards the right and up

        # Add table and ball to the scene
        self.play(Create(table), Create(ball))

        # Animate the ball's motion with collision handling
        self.play(self.ball_move(ball, velocity), run_time=10, rate_func=linear)

    def ball_move(self, ball, velocity):
        # Move the ball and check for collisions with the walls
        def update_ball(ball, dt):
            # Update ball position based on velocity
            ball.shift(velocity[0] * dt * RIGHT + velocity[1] * dt * UP)

            # Get the current position of the ball
            x, y, _ = ball.get_center()

            # Collision with left and right walls
            if abs(x) > 3:  # Half-width of table (table width = 6)
                velocity[0] *= -1  # Reverse the horizontal velocity

            # Collision with top and bottom walls
            if abs(y) > 1.5:  # Half-height of table (table height = 3)
                velocity[1] *= -1  # Reverse the vertical velocity

        # Use updater to move ball during the animation
        ball.add_updater(update_ball)

        return ball
