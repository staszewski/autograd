from manim import *
import numpy as np

# LLM generated - just for visuals
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
        self.wait(0.5)

        # ===== FORWARD PASS (Top Row) =====
        # Input node
        x_node = Circle(radius=0.3, color=BLUE, fill_opacity=0.1).move_to(LEFT * 4 + UP * 1)
        x_label = MathTex(r"\mathbf{x}", font_size=24, color=BLUE).move_to(x_node)
        x_group = VGroup(x_node, x_label)

        # Pre-activation node
        z_node = Circle(radius=0.3, color=GREEN_E, fill_opacity=0.1).move_to(LEFT * 1 + UP * 1)
        z_label = MathTex(r"\mathbf{z}", font_size=24, color=GREEN_E).move_to(z_node)
        z_group = VGroup(z_node, z_label)

        # Activation output node (layer output) - larger to fit label better
        h_node = Circle(radius=0.6, color=GREEN, fill_opacity=0.1).move_to(RIGHT * 2 + UP * 1)
        # Center h label
        h_label = MathTex(r"\mathbf{h}", font_size=20, color=GREEN).move_to(h_node.get_center())
        # Position sigma label below h_label, centered horizontally within the circle
        sigma_label = MathTex(r"= \sigma(\mathbf{z})", font_size=16, color=GREEN).move_to(h_node.get_center() + DOWN * 0.25)
        h_group = VGroup(h_node, h_label, sigma_label)

        # Forward arrows
        x_to_z_arrow = Arrow(x_node.get_right(), z_node.get_left(), color=WHITE, stroke_width=3)
        z_to_h_arrow = Arrow(z_node.get_right(), h_node.get_left(), color=WHITE, stroke_width=3)

        # Forward labels
        xz_label = MathTex(r"\mathbf{z} = W \mathbf{x} + \mathbf{b}", font_size=18).next_to(x_to_z_arrow, UP, buff=0.2)
        zh_label = MathTex(r"\sigma", font_size=18).next_to(z_to_h_arrow, UP, buff=0.2)  # Simplified label on arrow

        # Animate forward pass
        self.play(Create(x_group))
        self.play(Create(x_to_z_arrow), Write(xz_label))
        self.play(Create(z_group))
        self.play(Create(z_to_h_arrow), Write(zh_label))
        self.play(Create(h_group))
        self.wait(1)

        # ===== BACKWARD PASS (Bottom Row) =====
        # Gradient nodes (semi-transparent, positioned below)
        grad_y_node = Circle(radius=0.3, color=ORANGE, fill_opacity=0.1).move_to(h_node.get_center() + DOWN * 2.5)
        grad_y_label = MathTex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{h}}", font_size=18, color=ORANGE).move_to(grad_y_node)
        grad_y_group = VGroup(grad_y_node, grad_y_label)

        grad_z_node = Circle(radius=0.3, color=GREEN_E, fill_opacity=0.1).move_to(z_node.get_center() + DOWN * 2.5)
        grad_z_label = MathTex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{z}}", font_size=18, color=GREEN_E).move_to(grad_z_node)
        grad_z_group = VGroup(grad_z_node, grad_z_label)

        grad_x_node = Circle(radius=0.3, color=BLUE, fill_opacity=0.1).move_to(x_node.get_center() + DOWN * 2.5)
        grad_x_label = MathTex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{x}}", font_size=18, color=BLUE).move_to(grad_x_node)
        grad_x_group = VGroup(grad_x_node, grad_x_label)

        # Backward arrows (right to left, thicker for emphasis)
        y_to_z_arrow = Arrow(grad_y_node.get_left(), grad_z_node.get_right(), color=RED, stroke_width=4)
        z_to_x_arrow = Arrow(grad_z_node.get_left(), grad_x_node.get_right(), color=RED, stroke_width=4)

        # Key gradient labels (chain rule) - Positioned with more space
        # Activation derivative: ∂L/∂z = (∂L/∂h) ⊙ σ'(z)
        activation_chain = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \right) \odot \sigma'(\mathbf{z})",
            font_size=20, color=RED
        ).next_to(y_to_z_arrow, DOWN, buff=0.3)  # Increased buff for space
        # Use get_part_by_tex for robust highlighting of σ'(z)
        sigma_part = activation_chain.get_part_by_tex(r"\sigma'(\mathbf{z})")
        sigma_prime_rect = SurroundingRectangle(sigma_part, color=YELLOW, buff=0.15, stroke_width=2)
        deriv_label = Text("Activation Derivative (scales signal)", font_size=14, color=YELLOW).next_to(sigma_prime_rect, DOWN, buff=0.2)

        # Vanishing note - Position below the equation, centered
        vanishing_note = Text("If σ'(z) = 0 → gradient vanishes!", font_size=14, color=RED).move_to(activation_chain.get_center() + DOWN * 0.9)

        # Input chain: ∂L/∂x = W^T ∂L/∂z
        input_chain = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^\top \frac{\partial \mathcal{L}}{\partial \mathbf{z}}",
            font_size=20, color=RED
        ).next_to(z_to_x_arrow, DOWN, buff=0.3)  # Increased buff

        # Example derivative - Position below vanishing_note
        example_deriv = MathTex(r"\text{e.g., sigmoid: } \sigma'(z) = \sigma(z) (1 - \sigma(z))", font_size=16, color=GRAY).move_to(vanishing_note.get_center() + DOWN * 0.4)

        # Animate backward pass
        self.play(
            Create(grad_y_group), Create(grad_z_group), Create(grad_x_group),
            run_time=1.5
        )
        self.play(Create(y_to_z_arrow), Write(activation_chain))
        self.play(Create(sigma_prime_rect), Write(deriv_label))
        self.play(Write(vanishing_note))
        self.play(Write(example_deriv))
        self.wait(0.5)
        self.play(Create(z_to_x_arrow), Write(input_chain))
        self.wait(1)

        # Final fade out
        all_elements = VGroup(
            title, x_group, x_to_z_arrow, xz_label, z_group, z_to_h_arrow, zh_label, h_group,
            grad_y_group, grad_z_group, grad_x_group, y_to_z_arrow, activation_chain, sigma_prime_rect, deriv_label,
            vanishing_note, example_deriv, z_to_x_arrow, input_chain
        )
        self.play(FadeOut(all_elements), run_time=1)

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
        
        # Add performance comparison in top-right corner
        comparison_title = Text("Performance Comparison", font_size=18, color=WHITE).to_edge(UP).to_edge(RIGHT).shift(LEFT * 0.5 + DOWN * 0.5)
        self.play(Write(comparison_title))
        
        comparison = VGroup(
            Text("• ReLU: Fast convergence, no saturation", font_size=14, color=GREEN),
            Text("• Sigmoid: Slow convergence, vanishing gradients", font_size=14, color=YELLOW),
            Text("• Tanh: Moderate speed, zero-centered", font_size=14, color=PURPLE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(comparison_title, DOWN, buff=0.3).to_edge(RIGHT).shift(LEFT * 0.5)
        
        for item in comparison:
            self.play(Write(item))
            self.wait(0.5)
        
        self.wait(3)
        
        # Fade out
        all_elements = VGroup(title, axes, axes_labels, relu_curve, sigmoid_curve, tanh_curve,
                             legend, comparison_title, comparison)
        self.play(FadeOut(all_elements))