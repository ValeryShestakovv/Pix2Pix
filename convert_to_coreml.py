import torch
from generator_model import Generator
import config
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.neural_network import quantization_utils

def save_model_quantization():
    #квантизация модели при помощи инструментов coremltools
    print("start")
    model_fp32 = ct.models.MLModel('model_coreml.mlmodel')
    model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
    model_fp16.save("model_coreml_fp16.mlmodel")
    print("end")

def convert_to_coreml():
    # В PyTorch обычной практикой является подготовка данных перед передачей их в сеть, поэтому сеть не содержит
    # уровня нормализации. Но в CoreML и Vision такой функциональности нет. Вы должны либо самостоятельно подготовить
    # массив пикселей из UIImage и передать его непосредственно в CoreML (без использования Vision framework),
    # либо добавить уровень нормализации в свою сеть.
    # поскольку модель работает со значениями в диапазоне (-1, 1), а UIImage изображения в swift хранятся
    # в (0, 256), то необходимо нормализовать входные и выходные данные для модели coreml. С помощью
    # coremltools мы можем, почему-то, изменить только входные данные(реализация на 31-36 строках), но не можем
    # изменить выходные. Поэтому подгружаем измененную генеративную модель, отличие которой лишь в том, что
    # в послежнем слое нормализуем до (0, 256)
    model = Generator(in_channels=3, features=64).to(config.DEVICE)
    # вытаскивает из чекпоинта стейты генеративной модели
    checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    # создаем рандомный импут нужной размерности для трассировки модели
    example_input = torch.randn((1, 3, 256, 256))
    model.eval()
    traced_model = torch.jit.trace(model, example_input)

    #При помощи ImageType изменяем входные данные для модели
    scale = 1/(0.5*255.0)
    bias = [- 0.5 / 0.5, - 0.5 / 0.5, - 0.5 / 0.5]

    image_input = ct.ImageType(name="input_1",
                               shape=example_input.shape,
                               scale=scale, bias=bias)

    #конвертируем модель
    model = ct.convert(
        traced_model,
        inputs=[image_input],
    )

    model.save("model_coreml.mlmodel")
    #Указываем, что выходной слой выдает изображение размером 256 на 256
    spec = ct.utils.load_spec("model_coreml.mlmodel")

    output = spec.description.output[0]
    output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    output.type.imageType.height = 256
    output.type.imageType.width = 256

    ct.utils.save_spec(spec, "model_coreml.mlmodel")
    print(spec.description)

convert_to_coreml()