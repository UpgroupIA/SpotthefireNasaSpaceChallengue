import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 400, 300
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: ALTO")
  if answer == 1:
    print("pred: BAJO")

  return answer

predict('./data/entrenamiento/ALTO/alta1.jpg') 
predict('./data/entrenamiento/ALTO/alta2.jpg')
predict('./data/entrenamiento/ALTO/alta3.jpg')
predict('./data/entrenamiento/ALTO/alta4.jpg')
predict('./data/entrenamiento/ALTO/alta5.jpg')
predict('./data/entrenamiento/ALTO/alta6.jpg')
predict('./data/entrenamiento/ALTO/alta7.jpg')
predict('./data/entrenamiento/ALTO/alta8.jpg')
predict('./data/entrenamiento/ALTO/alta9.jpg')
predict('./data/entrenamiento/ALTO/alta10.jpg')
predict('./data/entrenamiento/ALTO/alta11.jpg')
predict('./data/entrenamiento/ALTO/alta12.jpg')
predict('./data/entrenamiento/ALTO/alta13.jpg')
predict('./data/entrenamiento/ALTO/alta14.jpg')
predict('./data/entrenamiento/ALTO/alta15.jpg')
predict('./data/entrenamiento/ALTO/alta16.jpg')
predict('./data/entrenamiento/ALTO/alta17.jpg')
predict('./data/entrenamiento/ALTO/alta18.jpg')
predict('./data/entrenamiento/ALTO/alta19.jpg')
predict('./data/entrenamiento/ALTO/alta20.jpg')
predict('./data/entrenamiento/ALTO/alta21.jpg')
predict('./data/entrenamiento/ALTO/alta22.jpg')
predict('./data/entrenamiento/ALTO/alta23.jpg')
predict('./data/entrenamiento/ALTO/alta24.jpg')
predict('./data/entrenamiento/ALTO/alta25.jpg')
predict('./data/entrenamiento/ALTO/alta26.jpg')

print("_______________________________________________________")

predict('./data/entrenamiento/BAJO/Bajo1.jpg')
predict('./data/entrenamiento/BAJO/Bajo2.jpg')
predict('./data/entrenamiento/BAJO/Bajo3.jpg')
predict('./data/entrenamiento/BAJO/Bajo4.jpg')
predict('./data/entrenamiento/BAJO/Bajo5.jpg')
predict('./data/entrenamiento/BAJO/Bajo6.jpg')
predict('./data/entrenamiento/BAJO/Bajo7.jpg')
predict('./data/entrenamiento/BAJO/Bajo8.jpg')
predict('./data/entrenamiento/BAJO/Bajo9.jpg')
predict('./data/entrenamiento/BAJO/Bajo10.jpg')
predict('./data/entrenamiento/BAJO/Bajo11.jpg')
predict('./data/entrenamiento/BAJO/Bajo12.jpg')
predict('./data/entrenamiento/BAJO/Bajo13.jpg')
predict('./data/entrenamiento/BAJO/Bajo14.jpg')
predict('./data/entrenamiento/BAJO/Bajo15.jpg')
predict('./data/entrenamiento/BAJO/Bajo16.jpg')
predict('./data/entrenamiento/BAJO/Bajo17.jpg')
predict('./data/entrenamiento/BAJO/Bajo18.jpg')
predict('./data/entrenamiento/BAJO/Bajo19.jpg')
predict('./data/entrenamiento/BAJO/Bajo20.jpg')
predict('./data/entrenamiento/BAJO/Bajo21.jpg')
predict('./data/entrenamiento/BAJO/Bajo22.jpg')
predict('./data/entrenamiento/BAJO/Bajo23.jpg')
predict('./data/entrenamiento/BAJO/Bajo24.jpg')
predict('./data/entrenamiento/BAJO/Bajo25.jpg')
predict('./data/entrenamiento/BAJO/Bajo26.jpg')
predict('./data/entrenamiento/BAJO/Bajo27.jpg')
predict('./data/entrenamiento/BAJO/Bajo28.jpg')
predict('./data/entrenamiento/BAJO/Bajo29.jpg')
predict('./data/entrenamiento/BAJO/Bajo30.jpg')
predict('./data/entrenamiento/BAJO/Bajo31.jpg')
predict('./data/entrenamiento/BAJO/Bajo32.jpg')
predict('./data/entrenamiento/BAJO/Bajo33.jpg')
predict('./data/entrenamiento/BAJO/Bajo34.jpg')
predict('./data/entrenamiento/BAJO/Bajo35.jpg')

