import tfcoreml as tf_converter
import numpy as np
from tensorflow.python.tools.freeze_graph import freeze_graph
from keras.preprocessing.image import load_img
import os
import tfcoreml
import coremltools
import posenet
from posenet.model import model_id_to_ord, load_config
from posenet.converter.config import load_config

MODEL_DIR = './_models'


def convert2(model_id, model_dir=MODEL_DIR):
    model_ord = model_id_to_ord(model_id)
    cfg = load_config()
    print(cfg)
    checkpoints = cfg['checkpoints']
    print(checkpoints)
    imageSize = cfg['imageSize']
    print(imageSize)
    chkpoint = checkpoints[model_ord]


    width = imageSize
    height = imageSize

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Provide these to run freeze_graph:
    # Graph definition file, stored as protobuf TEXT
    #graph_def_file = './models/model.pbtxt'
    # Frozen model's output name
    frozen_model_file = os.path.join(model_dir, "model-%s.pb" % chkpoint)
    coreml_model_file = os.path.join(model_dir, "model-%s.mlmodel" % chkpoint)
    # Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
    output_node_names = 'heatmap,offset_2,displacement_fwd_2,displacement_bwd_2'
    # output_node_names = 'Softmax' 
    input_tensor_shapes = {"image:0":[1, imageSize, imageSize, 3]} 
    # output_tensor_names = ['output:0']
    output_tensor_names = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

    coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file, 
        mlmodel_path=coreml_model_file, 
        input_name_shape_dict=input_tensor_shapes,
        image_input_names=['image:0'],
        output_feature_names=output_tensor_names,
        is_bgr=False,
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2./255)
       # use_coreml_3=True,)


    coreml_model.author = 'joker2017'
    coreml_model.license = 'MIT'
    coreml_model.short_description = 'Ver.0.0.1'

    coreml_model.save(model_dir + '/posenet'+ str(imageSize) + '_' + chkpoint +'.mlmodel')
    print("Model for CoreML saving")
#image = 'https://instagram.fhel6-1.fna.fbcdn.net/vp/29600a9e592d04b6cdd649c5a7ebbee1/5E53949F/t51.2885-15/sh0.08/e35/p640x640/25008636_1075666015908842_2177456931973627904_n.jpg?_nc_ht=instagram.fhel6-1.fna.fbcdn.net&_nc_cat=102'
#img = load_img("./images/tennis_in_crowd.jpg", target_size=(imageSize, imageSize))
#print(img)
#out = coreml_model.predict({'image__0': img})['heatmap__0']
#print("#output coreml result.")

#print(out.shape)
#print(np.transpose(out))
#print(out)
# print(out[:, 0:1, 0:1])
#print(np.mean(out))
