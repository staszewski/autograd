from manim import *
import numpy as np

# Configure Manim to use simple text instead of LaTeX
config.tex_template = "simple"

class ActivationFunctionsScene(Scene):
    def construct(self):
        # Title
        title = Text("Activation Functions", font_size=32, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes for function plots (simplified to avoid LaTeX)
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 0.5],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE},
        ).scale(0.8).shift(DOWN * 0.5)
        
        # Add labels
        func_label = Text("Functions", font_size=20, color=BLUE).next_to(axes, UP)
        
        self.play(Create(axes))
        self.wait(1)
        
        # Define activation functions
        def relu_func(x):
            return max(0, x)
        
        def sigmoid_func(x):
            return 1 / (1 + np.exp(-x))
        
        def tanh_func(x):
            return np.tanh(x)
        
        # Define derivatives
        def relu_deriv(x):
            return 1 if x > 0 else 0
        
        def sigmoid_deriv(x):
            s = sigmoid_func(x)
            return s * (1 - s)
        
        def tanh_deriv(x):
            return 1 - np.tanh(x)**2
        
        # Create function plots
        x_range = np.linspace(-4, 4, 1000)
        
        # ReLU
        relu_points = [axes.coords_to_point(x, relu_func(x)) for x in x_range]
        relu_graph = VMobject()
        relu_graph.set_points_as_corners(relu_points)
        relu_graph.set_color(GREEN)
        
        # Sigmoid
        sigmoid_points = [axes.coords_to_point(x, sigmoid_func(x)) for x in x_range]
        sigmoid_graph = VMobject()
        sigmoid_graph.set_points_as_corners(sigmoid_points)
        sigmoid_graph.set_color(YELLOW)
        
        # Tanh
        tanh_points = [axes.coords_to_point(x, tanh_func(x)) for x in x_range]
        tanh_graph = VMobject()
        tanh_graph.set_points_as_corners(tanh_points)
        tanh_graph.set_color(PURPLE)
        
        # Animate ReLU first
        relu_label = Text("ReLU: max(0, x)", font_size=16, color=GREEN).to_edge(LEFT).shift(UP * 2)
        
        self.play(Write(relu_label))
        self.play(Create(relu_graph))
        self.wait(2)
        
        # Animate Sigmoid
        sigmoid_label = MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", font_size=16, color=YELLOW).to_edge(LEFT).shift(UP * 1)
        
        self.play(Write(sigmoid_label))
        self.play(Create(sigmoid_graph))
        self.wait(2)
        
        # Animate Tanh
        tanh_label = MathTex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}", font_size=16, color=PURPLE).to_edge(LEFT)
        
        self.play(Write(tanh_label))
        self.play(Create(tanh_graph))
        self.wait(2)
        
        # Wait a moment to show all functions together
        self.wait(3)
        
        # Fade out everything
        all_elements = VGroup(title, axes, func_label, 
                             relu_graph, sigmoid_graph, tanh_graph,
                             relu_label, sigmoid_label, tanh_label)
        self.play(FadeOut(all_elements))


class ActivationDerivativesScene(Scene):
    def construct(self):
        # Title
        title = Text("Activation Function Derivatives", font_size=32, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes for derivative plots
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-0.5, 1.5, 0.25],
            x_length=10,
            y_length=6,
            axis_config={"color": RED},
        ).scale(0.8).shift(DOWN * 0.5)
        
        # Add labels
        deriv_label = Text("Derivatives", font_size=20, color=RED).next_to(axes, UP)
        
        self.play(Create(axes), Write(deriv_label))
        self.wait(1)
        
        # Define derivative functions
        def relu_deriv(x):
            return 1 if x > 0 else 0
        
        def sigmoid_deriv(x):
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        
        def tanh_deriv(x):
            return 1 - np.tanh(x)**2
        
        # Create derivative plots
        x_range = np.linspace(-4, 4, 1000)
        
        # ReLU derivative
        relu_deriv_points = [axes.coords_to_point(x, relu_deriv(x)) for x in x_range]
        relu_deriv_graph = VMobject()
        relu_deriv_graph.set_points_as_corners(relu_deriv_points)
        relu_deriv_graph.set_color(GREEN)
        
        # Sigmoid derivative
        sigmoid_deriv_points = [axes.coords_to_point(x, sigmoid_deriv(x)) for x in x_range]
        sigmoid_deriv_graph = VMobject()
        sigmoid_deriv_graph.set_points_as_corners(sigmoid_deriv_points)
        sigmoid_deriv_graph.set_color(YELLOW)
        
        # Tanh derivative
        tanh_deriv_points = [axes.coords_to_point(x, tanh_deriv(x)) for x in x_range]
        tanh_deriv_graph = VMobject()
        tanh_deriv_graph.set_points_as_corners(tanh_deriv_points)
        tanh_deriv_graph.set_color(PURPLE)
        
        # Animate ReLU derivative first
        relu_deriv_label = Text("ReLU': 1 if x > 0 else 0", font_size=16, color=GREEN).to_edge(LEFT).shift(UP * 2)
        
        self.play(Write(relu_deriv_label))
        self.play(Create(relu_deriv_graph))
        self.wait(2)
        
        # Animate Sigmoid derivative
        sigmoid_deriv_label = MathTex(r"\sigma'(x) = \sigma(x) (1 - \sigma(x))", font_size=16, color=YELLOW).to_edge(LEFT).shift(UP * 1)
        
        self.play(Write(sigmoid_deriv_label))
        self.play(Create(sigmoid_deriv_graph))
        self.wait(2)
        
        # Animate Tanh derivative
        tanh_deriv_label = MathTex(r"\tanh'(x) = 1 - \tanh^2(x)", font_size=16, color=PURPLE).to_edge(LEFT)
        
        self.play(Write(tanh_deriv_label))
        self.play(Create(tanh_deriv_graph))
        self.wait(2)
        
        # Highlight key properties
        self.play(
            FadeOut(relu_deriv_label),
            FadeOut(sigmoid_deriv_label),
            FadeOut(tanh_deriv_label)
        )
        
        # Key insights
        insights_title = Text("Derivative Properties", font_size=24, color=WHITE).to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(insights_title))
        
        insights = VGroup(
            Text("• ReLU': Step function (0 or 1)", font_size=16, color=GREEN),
            Text("• Sigmoid': Bell curve, peaks at x=0", font_size=16, color=YELLOW),
            Text("• Tanh': Bell curve, peaks at x=0", font_size=16, color=PURPLE),
            Text("• Vanishing gradient problem visible", font_size=16, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN * 1)
        
        for insight in insights:
            self.play(Write(insight))
            self.wait(1)
        
        self.wait(3)
        
        # Fade out everything
        all_elements = VGroup(title, axes, deriv_label, 
                             relu_deriv_graph, sigmoid_deriv_graph, tanh_deriv_graph, 
                             insights_title, insights)
        self.play(FadeOut(all_elements))


class GradientFlowScene(Scene):
    def construct(self):
        # Title
        title = Text("Gradient Flow Through Activation Functions", font_size=28, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create a simple neural network visualization
        # Input layer
        input_node = Circle(radius=0.3, color=BLUE).move_to(LEFT * 4)
        input_text = Text("x", font_size=20).move_to(input_node)
        input_group = VGroup(input_node, input_text)
        
        # Hidden layer
        hidden_node = Circle(radius=0.3, color=GREEN).move_to(ORIGIN)
        hidden_text = Text("h", font_size=20).move_to(hidden_node)
        hidden_group = VGroup(hidden_node, hidden_text)
        
        # Output layer
        output_node = Circle(radius=0.3, color=RED).move_to(RIGHT * 4)
        output_text = Text("y", font_size=20).move_to(output_node)
        output_group = VGroup(output_node, output_text)
        
        # Forward connections
        input_to_hidden = Arrow(input_node.get_right(), hidden_node.get_left(), color=WHITE, stroke_width=2)
        hidden_to_output = Arrow(hidden_node.get_right(), output_node.get_left(), color=WHITE, stroke_width=2)
        
        # Backward gradient arrows
        grad_arrow = Arrow(output_node.get_left(), hidden_node.get_right(), color=RED, stroke_width=3)
        grad_arrow.set_stroke(width=6)
        
        # Labels
        forward_label = MathTex(r"h = \sigma(Wx + b)", font_size=16).next_to(input_to_hidden, UP)
        backward_label = MathTex(r"\frac{\partial L}{\partial h} = \frac{\partial L}{\partial y} \times \sigma'(h)", font_size=16, color=RED).next_to(grad_arrow, DOWN)
        
        # Show the network
        self.play(Create(input_group))
        self.play(Create(input_to_hidden), Write(forward_label))
        self.play(Create(hidden_group))
        self.play(Create(hidden_to_output))
        self.play(Create(output_group))
        self.wait(1)
        
        # Show gradient flow
        self.play(Create(grad_arrow), Write(backward_label))
        self.wait(2)
        
        # Demonstrate different activation effects
        activation_title = Text("Activation Function Effects on Gradients", font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(activation_title))
        
        # Create comparison boxes
        relu_box = Rectangle(width=2, height=1.5, color=GREEN).move_to(LEFT * 2 + DOWN * 2)
        relu_text = Text("ReLU\nGradient = 1\n(if h > 0)", font_size=12, color=GREEN).move_to(relu_box)
        
        sigmoid_box = Rectangle(width=2, height=1.5, color=YELLOW).move_to(ORIGIN + DOWN * 2)
        sigmoid_text = Text("Sigmoid\nGradient = σ(1-σ)\n(0 < σ < 1)", font_size=12, color=YELLOW).move_to(sigmoid_box)
        
        tanh_box = Rectangle(width=2, height=1.5, color=PURPLE).move_to(RIGHT * 2 + DOWN * 2)
        tanh_text = Text("Tanh\nGradient = 1-tanh²\n(0 < 1-tanh² < 1)", font_size=12, color=PURPLE).move_to(tanh_box)
        
        self.play(
            Create(relu_box), Write(relu_text),
            Create(sigmoid_box), Write(sigmoid_text),
            Create(tanh_box), Write(tanh_text)
        )
        
        # Highlight vanishing gradient problem
        problem_text = Text("Vanishing Gradient Problem:", font_size=16, color=RED).to_edge(DOWN).shift(UP * 0.5)
        explanation = Text("Sigmoid/Tanh gradients → 0 as |x| → ∞", font_size=14, color=RED).next_to(problem_text, DOWN)
        
        self.play(Write(problem_text), Write(explanation))
        
        # Animate gradient strength
        for box, color in [(relu_box, GREEN), (sigmoid_box, YELLOW), (tanh_box, PURPLE)]:
            self.play(Indicate(box, color=color, scale_factor=1.2))
            self.wait(0.5)
        
        self.wait(3)
        
        # Fade out
        all_elements = VGroup(title, input_group, input_to_hidden, forward_label, hidden_group,
                             hidden_to_output, output_group, grad_arrow, backward_label,
                             activation_title, relu_box, relu_text, sigmoid_box, sigmoid_text,
                             tanh_box, tanh_text, problem_text, explanation)
        self.play(FadeOut(all_elements))


class LearningDynamicsScene(Scene):
    def construct(self):
        # Title
        title = Text("Learning Dynamics: Different Activation Functions", font_size=24, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes for loss curves (simplified to avoid LaTeX)
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 2, 0.5],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE},
        ).scale(0.8).shift(DOWN * 0.5)
        
        axes_labels = VGroup(
            Text("Epochs", font_size=16).next_to(axes.x_axis, DOWN),
            Text("Loss", font_size=16).next_to(axes.y_axis, LEFT).rotate(PI/2)
        )
        
        self.play(Create(axes), Write(axes_labels))
        
        # Simulate loss curves for different activations
        epochs = np.linspace(0, 100, 100)
        
        # ReLU: fast initial convergence, then steady
        relu_loss = 2 * np.exp(-epochs/30) + 0.1
        relu_points = [axes.coords_to_point(epoch, loss) for epoch, loss in zip(epochs, relu_loss)]
        relu_curve = VMobject()
        relu_curve.set_points_as_corners(relu_points)
        relu_curve.set_color(GREEN)
        
        # Sigmoid: slow convergence due to vanishing gradients
        sigmoid_loss = 1.8 * np.exp(-epochs/60) + 0.3
        sigmoid_points = [axes.coords_to_point(epoch, loss) for epoch, loss in zip(epochs, sigmoid_loss)]
        sigmoid_curve = VMobject()
        sigmoid_curve.set_points_as_corners(sigmoid_points)
        sigmoid_curve.set_color(YELLOW)
        
        # Tanh: moderate convergence
        tanh_loss = 1.9 * np.exp(-epochs/45) + 0.2
        tanh_points = [axes.coords_to_point(epoch, loss) for epoch, loss in zip(epochs, tanh_loss)]
        tanh_curve = VMobject()
        tanh_curve.set_points_as_corners(tanh_points)
        tanh_curve.set_color(PURPLE)
        
        # Legend
        legend = VGroup(
            Text("ReLU", font_size=16, color=GREEN).to_edge(LEFT).shift(UP * 2),
            Text("Sigmoid", font_size=16, color=YELLOW).to_edge(LEFT).shift(UP * 1.5),
            Text("Tanh", font_size=16, color=PURPLE).to_edge(LEFT).shift(UP * 1)
        )
        
        self.play(Write(legend))
        
        # Animate the curves
        self.play(Create(relu_curve), run_time=3)
        self.wait(1)
        self.play(Create(sigmoid_curve), run_time=3)
        self.wait(1)
        self.play(Create(tanh_curve), run_time=3)
        self.wait(2)
        
        # Add performance comparison
        comparison_title = Text("Performance Comparison", font_size=18, color=WHITE).to_edge(DOWN).shift(UP * 1)
        self.play(Write(comparison_title))
        
        comparison = VGroup(
            Text("• ReLU: Fast convergence, no saturation", font_size=14, color=GREEN),
            Text("• Sigmoid: Slow convergence, vanishing gradients", font_size=14, color=YELLOW),
            Text("• Tanh: Moderate speed, zero-centered", font_size=14, color=PURPLE)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN).shift(UP * 0.5)
        
        for item in comparison:
            self.play(Write(item))
            self.wait(0.5)
        
        self.wait(3)
        
        # Fade out
        all_elements = VGroup(title, axes, axes_labels, relu_curve, sigmoid_curve, tanh_curve,
                             legend, comparison_title, comparison)
        self.play(FadeOut(all_elements))