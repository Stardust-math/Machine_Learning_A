import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tk, filedialog
import copy
import json
import numpy as np
from PIL import Image
import tqdm

import math


class Algorithm:
    def __init__(
            self, r=0.5, mu=1, lambda_1=1e-2, lambda_2=1e-2, lambda_3=1e-2
    ) -> None:
        """
        Initialize the algorithm with given parameters.
        """

        # Initialize parameters
        self.r = r
        self.mu = mu
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        # Initialize matrices
        self.A: np.matrix = np.matrix(np.zeros((2, 2))) # shape (2, 2)
        self.b: np.matrix = np.matrix(np.zeros((2, 1))) # shape (2, 1)
        self.W: np.matrix = None

        # Training data
        self.X = None  # source points
        self.Y = None  # target points
        self.N = 0    # number of point pairs

    def rbf(self, x_1, x_2) -> float:
        """
        Radial Basis Function (RBF)
        φ_i(x) = (||x - x_i||^2 + r^2)^(μ/2)
        """
        x_1 = np.asarray(x_1)
        x_2 = np.asarray(x_2)
        return (np.linalg.norm(x_1 - x_2) ** 2 + self.r ** 2) ** (self.mu / 2)

    def train(self, point_pairs) -> None:
        """
        Train the algorithm using the provided point pairs.
        """

        # Check if point_pairs is empty
        if len(point_pairs) == 0:
            print("No point pairs provided for training.")
            return
        
        # Number of point pairs
        self.N = len(point_pairs)

        # Extract source and target points
        self.X = np.matrix([pair[1] for pair in point_pairs]).T # shape (2, N)
        self.Y = np.matrix([pair[0] for pair in point_pairs]).T # shape (2, N)

        # Construct the basis function matrix Phi ( N x N )
        Phi = np.matrix(np.zeros((self.N, self.N)))
        for i in range(self.N):
            for j in range(self.N):
                Phi[i, j] = self.rbf(self.X[:, i].T, self.X[:, j].T) # Phi[i, j] = (||x_i - x_j||^2 + r^2)^(mu/2)

        # Construct the extended matrix G (N x (3+N))
        ones = np.ones((self.N, 1)) # shape (N, 1)
        X_T = self.X.T # shape (N, 2)
        G = np.hstack([X_T, ones, Phi]) # shape (N, 3+N)

        # Construct the regularization matrix R ((3+N) x (3+N))
        R = np.zeros((3 + self.N, 3 + self.N))
        
        # Regularization for A (first 2x2 block)
        R[0:2, 0:2] = self.lambda_1 * np.eye(2)
        
        # Regularization for b (third diagonal element)
        R[2, 2] = self.lambda_2
        
        # Regularization for W (bottom-right NxN block)
        R[3:, 3:] = self.lambda_3 * np.eye(self.N)

        try:
            # Solve the linear system: (G^T G + R) theta = G^T Y^T
            left_matrix = G.T @ G + R
            right_matrix = G.T @ self.Y.T # shape (3+N, 2)
            
            # Solve for parameters theta = [A^T, b^T, W^T]^T
            theta = np.linalg.solve(left_matrix, right_matrix)
            
            # Extract parameters
            self.A = theta[0:2, :].T # shape (2, 2)
            self.b = theta[2:3, :].T # shape (2, 1)
            self.W = theta[3:, :].T  # shape (2, N)
            
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            left_matrix = G.T @ G + R
            right_matrix = G.T @ self.Y.T
            theta = np.linalg.pinv(left_matrix) @ right_matrix
            
            self.A = theta[0:2, :].T
            self.b = theta[2:3, :].T
            self.W = theta[3:, :].T

    def predict(self, x, y, point_pairs) -> tuple:
        """
        Predict the transformed coordinates for the input point (x, y).
        """

        # If not trained, return the input point
        if self.X is None or self.N == 0:
            return x, y
        
        # Prepare input point
        input_point = np.matrix([x, y]).T

        # Compute the RBF vector φ(x)
        phi_vec = np.matrix(np.zeros((self.N, 1)))
        for i in range(self.N):
            phi_vec[i, 0] = self.rbf([x, y], self.X[:, i].T.A1)

        # Compute the predicted point
        predicted = self.A * input_point + self.b + self.W * phi_vec

        # Extract predicted coordinates
        predicted_x = float(predicted[0, 0])
        predicted_y = float(predicted[1, 0])

        return predicted_x, predicted_y  # example return


class Application:
    def __init__(self) -> None:
        self.point_pairs = []
        self.current_start_point = None
        self.current_end_point = None
        self.is_mouse_pressed = False
        self.img_left = None
        self.img_right = None

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)

        # mouse events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        # button events
        button_labels = [
            "Clear",
            "Run Algorithm",
            "Load Image",
            "Save Image",
            "Load Data",
            "Save Data",
        ]
        button_callbacks = [
            self.clear_points,
            self.run_algorithm,
            self.load_image,
            self.save_image,
            self.load_json,
            self.save_json,
        ]

        self.buttons = []
        self.create_buttons(button_labels, button_callbacks)

    def clear_points(self, event):
        self.point_pairs = []
        self.on_frame()
        self.fig.canvas.draw()

    def create_buttons(self, button_labels, button_callbacks):
        num_buttons = len(button_labels)
        button_width = 0.1
        button_height = 0.075
        spacing = 0.05

        total_width = num_buttons * button_width + (num_buttons - 1) * spacing
        start_x = (1 - total_width) / 2  # 计算起始位置

        for i, label in enumerate(button_labels):
            x_position = start_x + i * (button_width + spacing)
            ax_button = plt.axes([x_position, 0.05, button_width, button_height])
            button = Button(ax_button, label)
            self.buttons.append(button)
            button.on_clicked(button_callbacks[i])

    def on_frame(self) -> None:
        self.ax_left.clear()
        if self.img_left is not None:
            self.ax_left.imshow(self.img_left)
        print("on_frame: num of points = ", len(self.point_pairs))
        for point_pair in self.point_pairs:
            start, end = point_pair
            self.ax_left.plot([start[0], end[0]], [start[1], end[1]], "b", alpha=0.8)
            self.ax_left.scatter([start[0]], [start[1]], c=["r"], alpha=0.8)
            self.ax_left.scatter([end[0]], [end[1]], c=["orange"], alpha=0.8)

    def run(self) -> None:
        plt.show()

    def on_press(self, event):
        if event.inaxes == self.ax_left and not self.is_mouse_pressed:
            self.is_mouse_pressed = True
            self.current_start_point = [event.xdata, event.ydata]
            self.current_end_point = [event.xdata, event.ydata]

    def on_release(self, event):
        if event.inaxes == self.ax_left:
            self.is_mouse_pressed = False
            self.current_end_point = [event.xdata, event.ydata]
            self.point_pairs.append([self.current_start_point, self.current_end_point])
            self.on_frame()
            self.fig.canvas.draw()

    def on_motion(self, event):
        if (
                event.inaxes == self.ax_left
                and self.current_start_point is not None
                and self.is_mouse_pressed
        ):
            self.current_end_point = [event.xdata, event.ydata]
            self.on_frame()
            start = self.current_start_point
            end = self.current_end_point
            self.ax_left.plot([start[0], end[0]], [start[1], end[1]], "r", alpha=0.8)
            self.ax_left.scatter([start[0]], [start[1]], c=["r"], alpha=0.8)
            self.ax_left.scatter([end[0]], [end[1]], c=["orange"], alpha=0.8)
            self.fig.canvas.draw()

    def save_json(self, event) -> None:
        root = Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
            title="Save Data As",
        )
        with open(file_path, "w") as file:
            json.dump(self.point_pairs, file)

    def load_json(self, event) -> None:
        root = Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(
            title="Select an JSON file",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        try:
            with open(file_path, "r") as file:
                self.point_pairs = json.load(file)
            print(f"Successfully loaded file {file_path}.")
            self.on_frame()
            self.fig.canvas.draw()
        except FileNotFoundError:
            print(f"File {file_path} not found!")

    def run_algorithm(self, event) -> None:
        if self.img_left is None:
            print("Load an image first!")
            return
        algorithm = Algorithm()
        algorithm.train(self.point_pairs)
        self.img_right = np.zeros_like(self.img_left)
        x_max, y_max, _ = self.img_right.shape
        for x in tqdm.tqdm(range(x_max)):
            for y in range(y_max):
                x_src, y_src = algorithm.predict(x, y, self.point_pairs)
                x_src_int = int(round(x_src))
                y_src_int = int(round(y_src))
                if 0 <= x_src_int < x_max and 0 <= y_src_int < y_max:
                    print(f"Predicted ({x}, {y}) -> ({x_src_int}, {y_src_int})")
                    self.img_right[x][y] = self.img_left[x_src_int][y_src_int]
        self.ax_right.clear()
        self.ax_right.imshow(self.img_right, cmap="gray")
        self.fig.canvas.draw()

    def save_image(self, event) -> None:
        if self.img_right is None:
            print("No image generated!")
            return
        root = Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
            title="Save Image As",
        )
        if file_path:
            image = Image.fromarray(self.img_right)
            image.save(file_path)

    def load_image(self, event) -> None:
        root = Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("All files", "*.*")],
        )
        if file_path:
            print(f"Loading image from: {file_path}")
            img = Image.open(file_path)
            img = img.convert("RGB")

            self.img_left = np.array(img)
            self.on_frame()
            self.fig.canvas.draw()


if __name__ == "__main__":
    app = Application()
    app.run()
