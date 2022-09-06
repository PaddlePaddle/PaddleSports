from paddle import inference
import os


def create_predictor(model_dir):
    # refer   https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_doc/Config/GPUConfig.html
    model_file = os.path.join(model_dir,'.pdmodel')
    params_file = os.path.join(model_dir,'.pdiparams')
    config = inference.Config()
    config.set_prog_file(model_file)
    config.set_params_file(params_file)
    # 启用 GPU 进行预测 - 初始化 GPU 显存 50M, Deivce_ID 为 0
    config.enable_use_gpu(50, 0)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handles = []
    for input_name in input_names:
        input_handles.append(predictor.get_input_handle(input_name))
    output_names = predictor.get_output_names()
    output_handles = []
    for output_name in output_names:
        output_handles.append(predictor.get_input_handle(output_name))
    print('model {} has {} inputs tensor {} outputs tensor'.format(model_dir, len(input_names), len(output_names)))
    return predictor, input_handles, output_handles


def create_predictor_un_combined(model_dir):
    # 加载非Combined 模型
    config = inference.Config()
    config.set_model(model_dir)
    config.enable_use_gpu(50, 0)
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handles = []
    for input_name in input_names:
        input_handles.append(predictor.get_input_handle(input_name))
    output_names = predictor.get_output_names()
    output_handles = []
    for output_name in output_names:
        output_handles.append(predictor.get_input_handle(output_name))
    print('model {} has {} inputs tensor {} outputs tensor'.format(model_dir, len(input_names), len(output_names)))
    return predictor, input_handles, output_handles