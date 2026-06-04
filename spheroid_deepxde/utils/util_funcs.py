def get_layer_size(i, h, n, o):
    return [i] + [n] * h + [o]

def save_model(model, dir_path, file_name):
    save_path = f'{dir_path}/{file_name}.pth'
    model.save(save_path)