import numpy as np
from abc import ABC, abstractmethod, ABCMeta
import abc

# Abstract Class for Objective Function
class ObjectiveFunction(ABC):

    @property
    @abstractmethod
    def functionBoundary(self):
        pass

    @property
    @abstractmethod
    def nDimension(self):
        pass

    @property
    @abstractmethod
    def optimalX(self):
        pass

    @property
    @abstractmethod
    def optimalF(self):
        pass

    @property
    @abstractmethod
    def inputX(self):
        pass

    @property
    @abstractmethod
    def functionName(self):
        pass

    @property
    @abstractmethod
    def functionEquationLaTeX(self):
        pass

    @property
    @abstractmethod
    def fuctionMultiModal(self):
        pass

    @property
    @abstractmethod
    def functionCode(self):
        pass

    @abstractmethod
    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        pass

class Ackley(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-32.768, 32.768]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def functionName(self):
        return "Ackley"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) = -a \cdot exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2})-exp(\frac{1}{d}\sum_{i=1}^{d}cos(c \cdot x_i))+ a + exp(1)"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F10"

    @property
    def inputX(self):
        return self._inputX

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Ackley function is widely used for testing optimization algorithms.
        In its two-dimensional form, it is characterized by a nearly flat outer region, and a large hole at the centre.
        The function poses a risk for optimization algorithms, particularly hillclimbing algorithms,
        to be trapped in one of its many local minima.
        Recommended variable values are: a = 20, b = 0.2 and c = 2π.
        The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768], for all i = 1, …, d,
        although it may also be restricted to a smaller domain.
        Reference: http://www.sfu.ca/~ssurjano/ackley.html"""
        self._inputX = x
        self._nDimension = len(x)

        a = 20
        b = 0.2
        c = 2 * np.pi

        dinverse = 1 / self._nDimension
        return a + np.exp(1) - (a * np.exp(-b * np.sqrt(dinverse * np.sum(self._inputX ** 2)))) - np.exp(
            dinverse * np.sum(np.cos(c * self._inputX)))

class Griewank(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-600, 600]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def functionName(self):
        return "Griewank"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \frac{1}{4000} \sum_{i=1}^{d}{x_i^{2}} - \prod_{i=1}^{d}cos(\frac{x_i}{\sqrt{i}}) + 1"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F13"

    @property
    def inputX(self):
        return self._inputX

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Griewank function has many widespread local minima, which are regularly distributed.
        The complexity is shown in the zoomed-in plots.
        The function is usually evaluated on the hypercube xi ∈ [-600, 600], for all i = 1, …, d.
        Reference: http://www.sfu.ca/~ssurjano/griewank.html"""
        self._inputX = x
        self._nDimension = len(x)

        prodPart = 1
        for i in range(self._nDimension):
            prodPart = prodPart * (np.cos(self._inputX[i] / np.sqrt(i + 1)))
        return (np.sum(self._inputX ** 2) / 4000) - prodPart + 1

class Levy(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-10, 10]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 1

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Levy"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F14"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &=\sin^2(\pi w_1)+\sum_{i=1}^{d-1}(w_i-1)^2 [1+10\sin^2(\pi w_i+1)]+(w_d-1)^2 [1+\sin^2(2\pi w_d)] \newline w_i=1+\frac{x_i-1}{4}: \forall i=1, ..., d"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
        Reference: http://www.sfu.ca/~ssurjano/levy.html"""
        self._inputX = x
        self._nDimension = len(x)

        w = 1 + (self._inputX - 1) / 4

        w1 = w[0]
        wd = w[-1]

        p1 = np.power(np.sin(np.pi * w1), 2)
        p2 = np.sum(((w - 1) ** 2) * (1 + (10 * np.sin(np.pi * w + 1)) ** 2))
        p3 = ((wd - 1) ** 2) * (1 + (np.sin(2 * np.pi * wd)) ** 2)

        return p1 + p2 + p3

class Rastrigin(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-5.12, 5.12]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Rastrigin"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F16"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x})&=\sum_{i=1}^{d}(x_i^2 - 10cos(2\pi x_i) + 10)"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Rastrigin function has several local minima. It is highly multimodal,
                but locations of the minima are regularly distributed. The function is usually evaluated on the hypercube
                xi ∈ [-5.12, 5.12], for all i = 1, …, dim. Global minimum is 0 at x=(0,...,0).
                Reference: https://www.sfu.ca/~ssurjano/rastr.html """
        self._inputX = x
        self._nDimension = len(x)
        return np.sum((self._inputX ** 2) - (10 * np.cos(2 * np.pi * self._inputX))) + (10 * self._nDimension)

class Schwefel(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-500, 500]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 420.9687 # self._nDimension * 420.9687

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Schwefel"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F17"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= 418.9829d -{\sum_{i=1}^{d} x_i sin(\sqrt{|x_i|})}"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Schwefel function is complex, with many local minima.
        The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, …, dim.
        Global minimum is 0 at x=(420.9687,...,420.9687).
        Reference: https://www.sfu.ca/~ssurjano/schwef.html"""
        self._inputX = x
        self._nDimension = len(x)

        alpha = 418.982887
        firstSum = np.sum(self._inputX * np.sin(np.sqrt(np.fabs(self._inputX))))
        return alpha * self._nDimension - firstSum

class Schwefel220(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-100, 100]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Schwefel 2.20"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F03"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^d |x_i|"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Schwefel function is complex, with many local minima.
        The function is usually evaluated on the hypercube xi ∈ [-100, 100], for all i = 1, …, dim.
        Global minimum is 0 at x=(0,...,0).
        Reference: https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12"""
        self._inputX = x
        self._nDimension = len(x)
        return np.sum(np.fabs(self._inputX))

class Schwefel221(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-100, 100]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Schwefel 2.21"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F04"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \max_{i \in \llbracket 1, d\rrbracket}|x_i|"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Schwefel function is complex, with many local minima.
        The function is usually evaluated on the hypercube xi ∈ [-100, 100], for all i = 1, …, dim.
        Global minimum is 0 at x=(0,...,0).
        Reference: https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12"""
        self._inputX = x
        self._nDimension = len(x)
        return np.max(np.fabs(self._inputX))

class Schwefel222(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-100, 100]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Schwefel 2.22"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F05"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^{d}|x_i|+\prod_{i=1}^{d}|x_i|"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Schwefel function is complex, with many local minima.
        The function is usually evaluated on the hypercube xi ∈ [-100, 100], for all i = 1, …, dim.
        Global minimum is 0 at x=(0,...,0).
        Reference: https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12"""
        self._inputX = x
        self._nDimension = len(x)
        return np.sum(np.fabs(self._inputX)) + np.prod(np.fabs(self._inputX))

class Schwefel223(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-10, 10]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Schwefel 2.23"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F06"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^{d}x_i^{10}"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Schwefel function is complex, with many local minima.
        The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, dim.
        Global minimum is 0 at x=(0,...,0).
        Reference: https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12"""
        self._inputX = x
        self._nDimension = len(x)
        return np.sum(self._inputX ** 10)

class Schwefel226(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-500, 500]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 420.968746 # np.ones(self._nDimension) * 420.968746

    @property
    def optimalF(self):
        return -418.982887272433799807913601398 * self._nDimension

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Schwefel 2.26"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F18"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x})&=-{\sum_{i=1}^{d} x_i sin(\sqrt{|x_i|})}"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The Schwefel function is complex, with many local minima.
        The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, …, dim.
        Global minimum is -418.982887272433799807913601398 * dimension at x=(420.968746,...,420.968746).
        Reference: https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/176-generalized-schwefel-s-problem-2-26"""
        self._inputX = x
        self._nDimension = len(x)
        return -np.sum(self._inputX * np.sin(np.sqrt(np.fabs(self._inputX))))

class Quartic(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-1.28, 1.28]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return (0 + np.random.normal(0, 1, 1)).item()

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Quartic"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F15"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x})&=\sum_{i=1}^{n}ix_i^4+\text{random}[0,1)"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima)
        Reference: https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12"""
        self._inputX = x
        self._nDimension = len(x)

        # Optimized Version from GitHub Lib:
        # d = X.shape[0]
        # res = np.sum(np.arange(1, d + 1) * X**4) + np.random.random()

        sum = 0
        for i in range(self._nDimension):
            sum += i * (self._inputX[i] ** 4)
        return sum + np.random.random()

class RotatedHyperEllipsoid(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-65.536, 65.536]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "RotatedHyperEllipsoid"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F02"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^{d}\sum_{j=1}^{i}x_j^2"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Bowl-Shaped) The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
        It is an extension of the Axis Parallel Hyper-Ellipsoid function, also referred to as the Sum Squares function.
        The plot shows its two-dimensional form.
        The function is usually evaluated on the hypercube xi ∈ [-65.536, 65.536], for all i = 1, …, d.
        Reference: http://www.sfu.ca/~ssurjano/rothyp.html"""
        self._inputX = x
        self._nDimension = len(x)

        sum = 0
        for i in range(self._nDimension):
            for j in range(i):
                sum += self._inputX[j] ** 2
        return sum

class Sphere(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-100, 100]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Sphere"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F07"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^{d} x_i^{2}"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Bowl-Shaped) The Sphere function is continuous, convex and unimodal. The function is usually evaluated on
        the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, dim. Global minimum is 0 at x=(0,...,0).
        Reference: https://www.sfu.ca/~ssurjano/spheref.html"""
        self._inputX = x
        self._nDimension = len(x)
        return np.sum(self._inputX ** 2)

class Step(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-100, 100]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return -0.5

    @property
    def optimalF(self):
        return 7.5

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Step"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F08"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^{d} (x_i^{2} + 0.5)"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Bowl-Shaped) The Sphere function is continuous, convex and unimodal. The function is usually evaluated on
        the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, dim. Global minimum is 0 at x=(0,...,0).
        Reference: https://www.sfu.ca/~ssurjano/spheref.html"""
        self._inputX = x
        self._nDimension = len(x)
        return np.sum(self._inputX ** 2 + 0.5)

class SumSquares(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-10, 10]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 0

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "SumSquares"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F09"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= \sum_{i=1}^{d}ix_i^{2}"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Bowl-Shaped) The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid function,
        has no local minimum except the global one. It is continuous, convex and unimodal. The function is usually
        evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, dim, although this may be restricted to the
        hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, dim. Global minimum is 0 at x=(0,...,0).
        Reference: https://www.sfu.ca/~ssurjano/sumsqu.html"""
        self._inputX = x
        self._nDimension = len(x)

        # Optimized Version from GitHub Lib
        # d = X.shape[0]
        # i = np.arange(1, d + 1)
        # res = np.sum(i * X**2)

        sum = 0
        for i in range(self._nDimension):
            sum += (i + 1) * (self._inputX[i] ** 2)
        return sum

class Rosenbrock(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-30, 30]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 1

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Rosenbrock"

    @property
    def fuctionMultiModal(self):
        return False

    @property
    def functionCode(self):
        return "F01"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x})=\sum_{i=1}^{d-1}[100 (x_{i+1} - x_i^2)^ 2 + (x_i-1)^2]"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Valley-Shaped) The Rosenbrock function, also referred to as the Valley or Banana function, is a popular
        test problem for gradient-based optimization algorithms.The function is unimodal, and the global minimum lies
        in a narrow, parabolic valley. However, even though this valley is easy to find, convergence to the minimum is
        difficult (Picheny et al., 2012). The function is usually evaluated on the hypercube xi ∈ [-5, 10],
        for all i = 1, …, dim, although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1, …, dim.
        Global minimum is 0 at x=(1,...,1).
        Reference: https://www.sfu.ca/~ssurjano/rosen.html"""
        self._inputX = x
        self._nDimension = len(x)

        sum = 0
        for i in range(self._nDimension):
            if i != self._nDimension - 1:
                sum += 100 * ((self._inputX[i + 1] - self._inputX[i] ** 2) ** 2) + ((self._inputX[i] - 1) ** 2)

        return sum

class GeneralizedPenalizedFunc1(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-50, 50]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return -1

    @property
    def optimalF(self):
        return 0

    @property
    def inputX(self):
        return self._inputX

    @property
    def functionName(self):
        return "Penalized 1"

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F11"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x})&=\frac{\pi}{d}\{10\sin(\pi w_1)+ \sum_{i=1}^{d-1}(w_i-1)^2[1+10\sin^2(\pi w_{i+1})] +(w_d-1)^2\} +\sum_{i=1}^du(x_i,10,100,4) \newline w_i=1+\frac{x_i+1}{4}: \forall i=1, ..., d"

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """() The function is usually evaluated on the hypercube xi ∈ [-50, 50],
        for all i = 1, …, dim.
        Global minimum is 0 at x=(-1,...,-1).
        Reference: https://www.al-roomi.org/benchmarks/unconstrained/n-dimensions/172-generalized-penalized-function-no-1"""
        self._inputX = x
        self._nDimension = len(x)

        a = 10
        k = 100
        m = 4

        xi = 1 + (self._inputX + 1) / 4
        x1 = xi[0]
        xd = xi[-1]

        p1 = 10 * np.power(np.sin(np.pi * x1), 2)

        p2 = 0
        for i in range(self._nDimension-1):
            p2 += ((xi[i] - 1) ** 2) * (1 + 10 * (np.power(np.sin(np.pi * xi[i+1]), 2)))

        p3 = ((xd - 1) ** 2)

        p4 = 0
        for i in range(self._nDimension):
            p4 += u(self._inputX[i], a, k, m)

        return ((np.pi / self._nDimension) * (p1 + p2 + p3)) + p4

class GeneralizedPenalizedFunc2(ObjectiveFunction):

    @property
    def functionBoundary(self):
        return [-50, 50]

    @property
    def nDimension(self):
        return len(self._inputX)

    @property
    def optimalX(self):
        return 1

    @property
    def optimalF(self):
        return 0

    @property
    def functionName(self):
        return "Penalized 2"

    @property
    def functionEquationLaTeX(self):
        return r"f(\mathbf{x}) &= 0.1 * [sin^2(3\pi x_1)+ \sum_{i=1}^{d}(x_i-1)^2[1+sin^2(3\pi x_i + 1)] +(x_d-1)^2[1+sin^2(2\pi x_d)]] + \sum_{i=1}^{d}u(x_i, 5, 100, 4) "

    @property
    def fuctionMultiModal(self):
        return True

    @property
    def functionCode(self):
        return "F12"

    @property
    def inputX(self):
        return self._inputX

    def functionCall(self, x, trainX=None, trainY=None, testX=None, testY=None):
        """(Many Local Minima) The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
        Reference: https://www.sciencedirect.com/science/article/abs/pii/S0965997816305646"""
        self._inputX = x
        self._nDimension = len(x)

        x1 = self._inputX[0]
        xd = self._inputX[-1]

        p1 = np.power(np.sin(3 * np.pi * x1), 2)
        p2 = np.sum(((self._inputX - 1) ** 2) * (1 + (np.sin(3 * np.pi * self._inputX + 1)) ** 2))
        p3 = ((xd - 1) ** 2) * (1 + (np.sin(2 * np.pi * xd)) ** 2)

        a = 5
        k = 100
        m = 4

        p4 = 0
        for i in range(self._nDimension):
            p4 += u(self._inputX[i], a, k, m)

        return (0.1 * (p1 + p2 + p3)) + p4

# ***************************************** Extra Functions ***********************************************

def u(x, a=5, k=100, m=4):
    if x > a:
        r = k * np.power((x - a), m)
    elif x < -a:
        r = k * np.power((-x - a), m)
    else:
        r = 0
    return r

# *********************************************************************************************************