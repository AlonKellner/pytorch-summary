import unittest
from torchsummary import summary, summary_string, InputSize
from torchsummary.tests.test_models.test_model import \
    SingleInputNet, \
    MultipleInputNet, \
    MultipleInputNetDifferentDtypes, \
    MultipleOutputNet, \
    ParameterReuseNet, ComplexInputNet
import torch

gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"


class torchsummaryTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        (total_params, trainable_params), result_size = summary(model, input, device="cpu", ignore=[SingleInputNet])
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)

    def test_parameter_reuse(self):
        model = ParameterReuseNet()
        input = 100
        (total_params, trainable_params), result_size = summary(model, input, device="cpu")
        self.assertEqual(total_params, 10100)
        self.assertEqual(trainable_params, 10100)

    def test_multiple_input(self):
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)
        (total_params, trainable_params), result_size = summary(
            model, [input1, input2], device="cpu")
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_complex_input(self):
        model = ComplexInputNet()
        input1 = (1, 300)
        input2 = (1, 300)
        (total_params, trainable_params), result_size = summary(
            model, InputSize({'x1': input1, 'x2': input2}), device="cpu")
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_multiple_output(self):
        model = MultipleOutputNet()
        input = (1, 300)
        (total_params, trainable_params), result_size = summary(
            model, input, device="cpu")
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_single_layer_network(self):
        model = torch.nn.Linear(2, 5)
        input = (1, 2)
        (total_params, trainable_params), result_size = summary(model, input, device="cpu")
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_single_layer_network_on_gpu(self):
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
        input = (1, 2)
        (total_params, trainable_params), result_size = summary(model, input, device=gpu_if_available)
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]
        (total_params, trainable_params), result_size = summary(
            model, [input1, input2], device="cpu", dtypes=dtypes)
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)


class torchsummarystringTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        result, (total_params, trainable_params), output_size = summary_string(
            model, input, device="cpu")
        self.assertEqual(type(result), str)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)


if __name__ == '__main__':
    unittest.main(buffer=True)
