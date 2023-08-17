import tensorflow as tf
import numpy as np
import os

## Replace {your-gcs-bucket} !!
BUCKET_ROOT='/gcs/vertex-test1-391714-bucket'

# Define variables
NUM_CLASSES = 5
EPOCHS=10
BATCH_SIZE = 32

IMG_HEIGHT = 180
IMG_WIDTH = 180

DATA_DIR = f'{BUCKET_ROOT}/flower_photos'
SAVE_MODEL_DIR = f'{BUCKET_ROOT}/multi-machine-output'

def create_datasets(data_dir, batch_size):
  '''Creates train and validation datasets.'''

  train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size)

  validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size)

  train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
  validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

  return train_dataset, validation_dataset


def create_model():
  '''Creates model.'''

  model = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
  ])
  return model

def _is_chief(task_type, task_id):
      '''Helper function. Determines if machine is chief.'''

  return task_type == 'chief'


def _get_temp_dir(dirpath, task_id):
      '''Helper function. Gets temporary directory for saving model.'''

  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id):
      '''Helper function. Gets filepath to save model.'''

  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

def main():  

  # Create distribution strategy
  # strategy = tf.distribute.MirroredStrategy()
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  # Get data
  GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
  train_dataset, validation_dataset = create_datasets(DATA_DIR, BATCH_SIZE)

  # Wrap model creation and compilation within scope of strategy
  with strategy.scope():
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

  history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
  )


  # Determine type and task of the machine from
  # the strategy cluster resolver
  task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)

  # Based on the type and task, write to the desired model path
  write_model_path = write_filepath(SAVE_MODEL_DIR, task_type, task_id)
  model.save(write_model_path)
  

if __name__ == "__main__":
    main()