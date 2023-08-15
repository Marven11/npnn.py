import functools
import inspect
import logging
import random
from abc import ABC, abstractmethod
from typing import Counter
from unittest import TestCase

import numpy as np
import torchvision

LEARNING_RATE = 3e-5
BATCH_NUM = 64
PRINT_EVERY_BATCH = 20
EPOCH = 20
IMG_SIZE = 28


def my_array(xs):
    return JustArray(np.array(xs))


def sum_of(xs):
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return xs[0] + xs[1]
    mid = len(xs) // 2
    return sum_of(xs[:mid]) + sum_of(xs[mid:])


class GraphNode(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, xs, g):
        pass

    def __repr__(self):
        params = list(inspect.signature(self.__init__).parameters.keys())
        values = {
            param: self.__dict__[param] for param in params if param in self.__dict__
        }
        return "{}({})".format(
            self.__class__.__name__, ", ".join([f"{k}={v}" for k, v in values.items()])
        )

    def exp(self):
        return ExpFunc(self)

    def relu(self):
        return ReluFunc(self)

    def __add__(self, other):
        return AddFunc(self, other)

    def __sub__(self, other):
        return SubFunc(self, other)

    def __mul__(self, other):
        return MulFunc(self, other)

    def __matmul__(self, other):
        return MatMulFunc(self, other)


class JustArray(GraphNode):
    def __init__(self, from_array):
        self.from_array = from_array

    def forward(self):
        return self.from_array

    def backward(self, xs, g):
        for x in xs:
            assert isinstance(x, JustArray)
        return [
            np.ones_like(x.from_array) * g if x is self else np.zeros_like(x.from_array)
            for x in xs
        ]

    def __repr__(self):
        return (
            f"JustArray(shape={repr(self.from_array.shape)}, id={id(self.from_array)})"
        )


class AddFunc(GraphNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @functools.lru_cache
    def forward(self):
        return self.a.forward() + self.b.forward()

    def backward(self, xs, g):
        return [
            ag + bg for ag, bg in zip(self.a.backward(xs, g), self.b.backward(xs, g))
        ]


class SubFunc(GraphNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @functools.lru_cache
    def forward(self):
        return self.a.forward() - self.b.forward()

    def backward(self, xs, g):
        return [
            ag - bg for ag, bg in zip(self.a.backward(xs, g), self.b.backward(xs, g))
        ]


class ExpFunc(GraphNode):
    def __init__(self, a):
        self.a = a

    @functools.lru_cache
    def forward(self):
        return np.exp(self.a.forward())

    def backward(self, xs, g):
        exp = self.forward()
        return [g * exp for g in self.a.backward(xs, g)]


class ReluFunc(GraphNode):
    def __init__(self, a):
        self.a = a

    @functools.lru_cache
    def forward(self):
        x = self.a.forward()
        return x * (x >= 0)

    def backward(self, xs, g):
        x = self.a.forward()
        return self.a.backward(xs, g * (x >= 0))


class MulFunc(GraphNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @functools.lru_cache
    def forward(self):
        return self.a.forward() * self.b.forward()

    def backward(self, xs, g):
        a, b = self.a.forward(), self.b.forward()
        return [
            ag + bg
            for ag, bg in zip(self.a.backward(xs, g * a), self.b.backward(xs, g * b))
        ]


class MatMulFunc(GraphNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @functools.lru_cache
    def forward(self):
        return self.a.forward() @ self.b.forward()

    def backward(self, xs, g):
        a, b = self.a.forward(), self.b.forward()
        assert len(a.shape) == 2
        assert len(b.shape) == 2
        df_da = np.stack([(g[i] * b).sum(1) for i in range(g.shape[0])])
        df_db = np.stack([(a.T * g[:, i]).sum(1) for i in range(g.shape[1])]).T
        da_dxs = self.a.backward(xs, df_da)
        db_dxs = self.b.backward(xs, df_db)
        return [da_dx + db_dx for da_dx, db_dx in zip(da_dxs, db_dxs)]


def linear(in_feat, out_feat):
    # W = JustArray(
    #     torch.nn.Linear(in_feat, out_feat).weight.detach().resize_((in_feat, out_feat)).numpy()
    # )
    W = JustArray(
        np.random.normal(
            scale=np.sqrt(2 / (in_feat + out_feat)), size=(in_feat, out_feat)
        )
    )
    B = JustArray(np.zeros((out_feat,)))

    def forward(input_x):
        return input_x @ W + B

    return forward, [W, B]


def relu():
    def forward(input_x):
        return input_x.relu()

    return forward, []


def sequential(*blocks):
    forward_funcs = []
    params = []

    for f, ps in blocks:
        forward_funcs.append(f)
        params += ps

    def forward(x):
        for f in forward_funcs:
            x = f(x)
        return x

    return forward, params


def mseloss(a, b):
    return (a - b) * (a - b)


def train_batch(net, params, batch):
    losses = []
    for img, tag in batch:
        y = JustArray(np.array([[1 if i == tag else 0 for i in range(10)]]))
        input_x = JustArray(img.reshape((1, IMG_SIZE * IMG_SIZE)))
        fx = net(input_x)
        loss = mseloss(y, fx)
        losses.append(loss)
    loss_sum = sum_of(losses)
    l_sum = loss_sum.forward()
    for p, g in zip(
        params, loss_sum.backward(params, np.ones_like(l_sum) / len(batch))
    ):
        g = g.sum(0)
        if g.mean() > 100000:
            continue
        p.from_array -= g * LEARNING_RATE
    return l_sum.mean()


def train(net, params):
    trainset, testset = get_dataloaders()
    trainset = list(trainset)
    for batch_num in range(0, len(trainset), BATCH_NUM):
        l_mean = train_batch(net, params, trainset[batch_num : batch_num + BATCH_NUM])
        if batch_num % (BATCH_NUM * PRINT_EVERY_BATCH) == 0:
            perc = round(batch_num / len(trainset) * 100)
            print(f"perc={perc}% loss_mean={l_mean:.4f}")


def test(net):
    trainset, testset = get_dataloaders()
    count = 0
    answers = []
    for img, tag in testset:
        input_x = JustArray(img.reshape((1, IMG_SIZE * IMG_SIZE)))
        fx = net(input_x)
        y_pred = fx.forward()
        answer = y_pred[0].argmax()
        if answer == tag:
            count += 1
        answers.append(answer)
    print(Counter(answers))
    return count


def get_dataloaders():
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize((14, 14), antialias=True)  #type: ignore
        ]
    )
    mnist_train = torchvision.datasets.FashionMNIST(
        root="/tmp/torchvision/data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="/tmp/torchvision/data", train=False, transform=trans, download=True
    )
    return [
        ((img.numpy(), tag) for img, tag in mnist_train),
        ((img.numpy(), tag) for img, tag in mnist_test),
    ]


class CommonTestCase(TestCase):
    def func_test(self, func, *xs):
        import torch

        torch_arrays = []
        for x in xs:
            x = torch.Tensor(x)
            x.requires_grad_(True)
            torch_arrays.append(x)
        torch_result = func(*torch_arrays)
        torch_result.sum().backward()
        torch_grads = [x.grad for x in torch_arrays]
        my_arrays = [my_array(x) for x in xs]
        my_func = func(*my_arrays)
        my_result = my_func.forward()
        my_grads = my_func.backward(my_arrays, np.ones_like(my_result))

        logging.info("Checking %s", repr(my_func))
        assert (torch_result.detach().numpy() - my_result < 1e2).all()
        for my_grad, torch_grad in zip(my_grads, torch_grads):
            assert (
                torch_grad.shape == my_grad.shape
            ), f"{torch_grad.shape} {my_grad.shape}"
            assert (torch_grad.detach().numpy() - my_grad < 1e2).all()

    def test_add(self):
        self.func_test(
            (lambda a, b, c: a + b + c + a),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 1, 4], [5, 1, 4], [1, 9, 1]],
            [[1, 9, 1], [1, 9, 8], [1, 0, 0]],
        )

    def test_sub(self):
        self.func_test(
            (lambda a, b, c: a - b - c + a),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 1, 4], [5, 1, 4], [1, 9, 1]],
            [[1, 9, 1], [1, 9, 8], [1, 0, 0]],
        )

    def test_exp(self):
        self.func_test(
            (lambda a, b, c: a.exp() + (b + c).exp() + a),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 1, 4], [5, 1, 4], [1, 9, 1]],
            [[1, 9, 1], [1, 9, 8], [1, 0, 0]],
        )

    def test_relu(self):
        self.func_test(
            (lambda a, b, c: a.relu() + (b - c * a).relu() + a),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 1, 4], [5, 1, 4], [1, 9, 1]],
            [[1, 9, 1], [1, 9, 8], [1, 0, 0]],
        )

    def test_mul(self):
        self.func_test(
            (lambda a, b, c: a * b + c * (a - b)),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 1, 4], [5, 1, 4], [1, 9, 1]],
            [[1, 9, 1], [1, 9, 8], [1, 0, 0]],
        )

    def test_matmul(self):
        self.func_test(
            (lambda a, b, c: a @ b + c @ a),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 1, 4], [5, 1, 4], [1, 9, 1]],
            [[1, 9, 1], [1, 9, 8], [1, 0, 0]],
        )

    def test_basic2(self):
        self.func_test(
            (lambda a, b, c, d: a * (b - c) + d),
            [[1, 2, 3], [1, 1, 4]],
            [[4, 5, 6], [5, 1, 4]],
            [[7, 8, 9], [1, 9, 1]],
            [[10, 11, 12], [9, 8, 10]],
        )

    def test_basic3(self):
        self.func_test(
            (lambda a, b, c, d: a @ b + c @ d),
            [[1, 1, 4], [5, 1, 4]],
            [[1, 9], [1, 9], [8, 10]],
            [[1, 1], [4, 5]],
            [[1, 4], [1, 9]],
        )

    def test_basic4(self):
        def test_func(y1, y2):
            return (y2 - y1) * (y2 - y1)

        self.func_test(
            test_func,
            [[2, 2]],
            [[1, 2]],
        )

    def test_basic5(self):
        def test_func(x, w, b, w2, b2, y):
            y1 = (x @ w + b).relu() @ w2 + b2
            return (y - y1) * (y - y1)

        self.func_test(
            test_func,
            [[1, 1, 4, 5, 1, 4]],
            [[1, 9, 1], [1, 9, 2], [8, 10, 3], [1, 9, 4], [1, 9, 5], [8, 10, 6]],
            [[1, 1, 4]],
            [[1, 2], [3, 4], [5, 6]],
            [[2, 2]],
            [[1, 2]],
        )

    def test_basic6(self):
        def test_func(x, w, b, w2, b2, y):
            y1 = (x @ w + b).relu() @ w2 + b2
            return (y - y1) * (y - y1)

        def random_size(size):
            if not size:
                return random.random()
            return [random_size(size[1:]) for _ in range(size[0])]

        for _ in range(100):
            self.func_test(
                test_func,
                random_size((1, 6)),
                random_size((6, 3)),
                random_size((1, 3)),
                random_size((3, 2)),
                random_size((1, 2)),
                random_size((1, 2)),
            )


def main():
    net, params = sequential(
        linear(IMG_SIZE * IMG_SIZE, 200),
        relu(),
        linear(200, 100),
        relu(),
        linear(100, 10),
    )
    print(f"correct={test(net)}")
    for epoch in range(1, EPOCH + 1):
        print(f"{epoch=}")
        train(net, params)
        print(f"correct={test(net)}")


if __name__ == "__main__":
    main()
