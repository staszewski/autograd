from manim import *

class ChainScene(Scene):
    def construct(self):
        # Nodes (manual positioning for horizontal chain with even spacing)
        x_node = Rectangle(width=1.5, height=0.8, color=BLUE).move_to(LEFT * 5)
        x_text = Text("x = 3", font_size=24).move_to(x_node.get_center())
        x_group = VGroup(x_node, x_text)

        y_node = Rectangle(width=1.5, height=0.8, color=BLUE).move_to(LEFT * 1.5)
        y_text = Text("y = 6", font_size=24).move_to(y_node.get_center())
        y_group = VGroup(y_node, y_text)

        z_node = Rectangle(width=1.5, height=0.8, color=BLUE).move_to(RIGHT * 1.5)
        z_text = Text("z = 7", font_size=24).move_to(z_node.get_center())
        z_group = VGroup(z_node, z_text)
        
        l_node = Rectangle(width=1.5, height=0.8, color=RED).move_to(RIGHT * 5)
        l_text = Text("L = 49", font_size=24).move_to(l_node.get_center())
        l_group = VGroup(l_node, l_text)
        
        # Forward edges with explicit equation labels
        xy_edge = Arrow(x_node.get_right(), y_node.get_left(), buff=0, color=BLACK, stroke_width=3)
        xy_label = Text("y = 2x", font_size=18).next_to(xy_edge.get_center(), UP, buff=0.2)
        xy_group = VGroup(xy_edge, xy_label)

        yz_edge = Arrow(y_node.get_right(), z_node.get_left(), buff=0, color=BLACK, stroke_width=3)
        yz_label = Text("z = y + 1", font_size=18).next_to(yz_edge.get_center(), UP, buff=0.2).shift(RIGHT * 0.1)  # Nudge right for centering
        yz_group = VGroup(yz_edge, yz_label)

        zl_edge = Arrow(z_node.get_right(), l_node.get_left(), buff=0, color=BLACK, stroke_width=3)
        zl_label = Text("L = z²", font_size=18).next_to(zl_edge.get_center(), UP, buff=0.2)
        zl_group = VGroup(zl_edge, zl_label)
        
        # Backward edges (DashedVMobject, direct connection, labels below center)
        lz_back = DashedVMobject(Arrow(l_node.get_left(), z_node.get_right(), buff=0, color=RED, stroke_width=3), num_dashes=8, dashed_ratio=0.6)
        lz_label_back = Text("∂L/∂z = 2z = 14", font_size=16, color=RED).next_to(lz_back.get_center(), DOWN, buff=0.2)
        lz_back_group = VGroup(lz_back, lz_label_back)

        zy_back = DashedVMobject(Arrow(z_node.get_left(), y_node.get_right(), buff=0, color=RED, stroke_width=3), num_dashes=8, dashed_ratio=0.6)
        zy_label_back = Text("∂z/∂y = 1", font_size=16, color=RED).next_to(zy_back.get_center(), DOWN, buff=0.2)
        zy_back_group = VGroup(zy_back, zy_label_back)

        yx_back = DashedVMobject(Arrow(y_node.get_left(), x_node.get_right(), buff=0, color=RED, stroke_width=3), num_dashes=8, dashed_ratio=0.6)
        yx_label_back = Text("∂y/∂x = 2", font_size=16, color=RED).next_to(yx_back.get_center(), DOWN, buff=0.2)
        yx_back_group = VGroup(yx_back, yx_label_back)
        
        # Total gradient arc (curved below, larger radius, label centered below)
        total_start = x_node.get_bottom() + DOWN * 0.5
        total_end = l_node.get_bottom() + DOWN * 0.5
        total_arc = CurvedArrow(total_start, total_end, angle=TAU / 4, radius=6, color=ORANGE, stroke_width=4)
        total_label = Text("dL/dx = 14 × 1 × 2 = 28", font_size=20, color=ORANGE).next_to(total_arc.get_center(), DOWN, buff=0.2)
        total_group = VGroup(total_arc, total_label)
        
        # Title
        title = Text("Chain Rule: Forward & Backward Flows", font_size=32, color=WHITE).to_edge(UP)
        
        # Collect all elements for fade-out
        all_elements = VGroup(title, x_group, xy_group, y_group, yz_group, z_group, zl_group, l_group,
                              lz_back_group, zy_back_group, yx_back_group, total_group)
        
        # Animation sequence
        self.play(Write(title))
        self.wait(0.5)
        
        # Forward pass
        self.play(Create(x_group))
        self.play(Create(xy_group))
        self.play(Create(y_group))
        self.play(Create(yz_group))
        self.play(Create(z_group))
        self.play(Create(zl_group))
        self.play(Create(l_group))
        self.wait(1)
        
        # Backward pass (fade in one by one)
        self.play(FadeIn(lz_back_group))
        self.wait(0.5)
        self.play(FadeIn(zy_back_group))
        self.wait(0.5)
        self.play(FadeIn(yx_back_group))
        self.wait(1)
        
        # Highlight total
        self.play(Create(total_group))
        self.wait(2)
        
        # Fade out
        self.play(FadeOut(all_elements))