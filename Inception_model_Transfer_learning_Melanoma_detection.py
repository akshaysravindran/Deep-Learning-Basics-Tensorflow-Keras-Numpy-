
"""
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%    Inception model Transfer Learning in Keras (Melanoma detection)     %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Summary: This script is used for transfer learning of Inception model for melanoma detection.
Here, I assess the impact of freezing different combination of layers


 Author: Akshay Sujatha Ravindran
 email: akshay dot s dot ravindran at gmail dot com
 Dec 4th 2017
"""



from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import  GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import SGD, adam, adagrad, rmsprop
from collections import Counter
from keras.losses import kullback_leibler_divergence, categorical_crossentropy,mean_squared_error
import os


batch_size_value = 32
column_dim       = 244
row_dim          = 244


# Helper: Save the model.

checkpointer = ModelCheckpoint(filepath=os.path.join('checkpoints', 'two_adam_inception.{epoch:03d}-{val_loss:.2f}.hdf5',),verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Data loaders
def get_generators():
    train_datagen = ImageDataGenerator(
        rescale            = 1./255,
        shear_range        =0.2,
        horizontal_flip    = True,
        rotation_range     = 10.,
        width_shift_range  = 0.2,
        height_shift_range = 0.2)

    test_datagen           = ImageDataGenerator(rescale=1./255)

    train_generator        = train_datagen.flow_from_directory(
        os.path.join('Train'),
        target_size        = (row_dim, column_dim),
        batch_size         = batch_size_value,
        class_mode         = 'categorical')

    validation_generator   = test_datagen.flow_from_directory(
        os.path.join('Validation'),
        target_size        = (row_dim, column_dim),
        batch_size         = batch_size_value,
        class_mode         = 'categorical')
    return train_generator, validation_generator


# Custom model if needed
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(row_dim, column_dim,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model


# Load the pretained model and update last layers
def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model  = InceptionV3(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x           = base_model.output
    x           = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x           = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model



def freeze_all_but_top(model,loss_type,optimisation_type,learning_rate):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
        
     # Hyper parameter search   
    if optimisation_type=="SGD":
            OPT=SGD
    elif optimisation_type=="ADAM":
        OPT=adam
    elif optimisation_type=="ADAGRAM":
        OPT=adagrad
        
    OPT=OPT(lr=learning_rate)            
    
    if loss_type=="KLD":
        LOSS='kullback_leibler_divergence'    
    elif loss_type=="CE":
        LOSS="categorical_crossentropy"
    elif loss_type=="MSE":
        LOSS="mean_squared_error"
        
         
         
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy', 'categorical_crossentropy'])

    return model


def freeze_all_but_mid_and_top(model,loss_type,optimisation_type,learning_rate):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
    # we need to recompile the model for these modifications to take effect
             
     # Hyper parameter search   
    if optimisation_type=="SGD":
            OPT=SGD
    elif optimisation_type=="ADAM":
        OPT=adam
    elif optimisation_type=="ADAGRAM":
        OPT=adagrad
    elif optimisation_type=="RMS":
         OPT=rmsprop
        
        
    OPT=OPT(lr=learning_rate)            
    
    if loss_type=="KLD":
        LOSS='kullback_leibler_divergence'    
    elif loss_type=="CE":
        LOSS="categorical_crossentropy"
    elif loss_type=="MSE":
        LOSS="mean_squared_error"
        
  # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy', 'categorical_crossentropy'])
    return model

def train_model(model, nb_epoch, train_generator,validation_generator,class_weight,callback):

    model.fit_generator(
        train_generator,
        steps_per_epoch=60,
        validation_data=validation_generator,
        validation_steps=5,
        epochs=nb_epoch,class_weight=class_weight,callbacks=callback,verbose=2)
    return model


def main():
    
    
     for Optimize in ["SGD","Adam","Adagrad","RMS"]:
         for loss in ["MSE","CE","KLD"]:
            for LR in [0.1,0.01,0.001,0.0001]:
                hparam = "Opt%s_LR_%s_loss%s" % (Optimize, LR, loss)    
                tensorboard = TensorBoard(log_dir='./logs/mid_twoclass+'+hparam, histogram_freq=1,
                                          write_graph=False, write_images=True)
                tensorboard1 = TensorBoard(log_dir='./logs/top_twoclass'+hparam, histogram_freq=1,
                                          write_graph=False, write_images=True)
                model = get_model()
                train_generator, validation_generator = get_generators()   
                
                counter = Counter(train_generator.classes)                          
                max_val = float(max(counter.values()))       
                class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
                
                model = freeze_all_but_top(model)
                model = train_model(model,10, train_generator,validation_generator, class_weights,[checkpointer, early_stopper, tensorboard1]) 
                model = freeze_all_but_mid_and_top(model)
                model = train_model(model,20, train_generator,validation_generator,class_weights, [checkpointer, early_stopper, tensorboard])   # main_model=build_model()
                model.save_weights('twoclass.h5')
    
if __name__ == '__main__':
    main()
