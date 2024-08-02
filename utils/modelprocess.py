import onnxsim
import argparse
import onnx


def simplify_onnx(onnx_model, output_path='simplified.onnx'):
    """
    simplify onnx model
        Args:
            onnx_model: onnx model
            output_path: output path
    """
    model_simplified, check = onnxsim.simplify(onnx_model)
    if check:
        print("The model simplification was successful.")
    else:
        print("The model simplification failed.")
    onnx.save(model_simplified, output_path)
    return model_simplified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='onnx_model.onnx')
    parser.add_argument('--output_path', type=str, default='simplified.onnx')
    args = parser.parse_args()
    onnx_model = onnx.load(args.onnx_path)
    onnx_model = simplify_onnx(onnx_model, args.output_path)
