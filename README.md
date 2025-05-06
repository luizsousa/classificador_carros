# Projeto de implementação de Modelo de Inteligência Artificial de Classifiacção de Carros em Estacionamentos

- Link para a pasta do Google Drive pública com os arquivos do projeto:
  
  https://drive.google.com/drive/folders/1W-Kl2wMPdOlSjDtKNQyNv6jwV9FEFrYJ?usp=drive_link


## Introdução

Este projeto teve como objetivo a criação de Modelos de Inteligência Artificial capazes de identificar carros parados em estacionamentos. Como objetivo secundário o projeto teve também um dataset de imagens de carros.

A motivação do projeto foi de aquisição de conhecimento prático em relação aos temas como processamento de vídeo, identificação e classificação de objetos, treinamento de redes neurais e fine tuning de modelos de classificação de imagem.

Na concepção do projeto foram utilizadas modelos YOLO para detecção de carros e placas e redes pré-treinadas da família EfficientNet para criação do modelo classificador.

Todas imagens utilizadas no projeto foram captadas em locais públicos sem utilização de imagens com direitos autorais ou de sites. A captação dos vídeos foi realizada utilizando uma câmera GoPro Hero 6.


## Escopo e Restrições dos Modelos

- Veículos Escolhidos

  O projeto ficou restrito a carros de passeio, excluindo caminhões, motos, vans e etc. E restrito aos modelos mais comuns encontrados em estacionamentos de Brasília.

- Posições

  Somente foi considerado as posições dianteiras e traseiras.

  No processamento das imagens existe um parâmetro que limita a angulação máxima do carro, excluindo imagens de imagens laterais.

  A ângulação é cálculada com base na distância do centro da placa em relação ao centro carro. Quando mais a placa está próxima da borda do carro mais lateral é a imagem.

- Classes

  Nos datasets existem 190 classes distintas na posição dianteira e 158 classes distintas na posição traseira.

	Os modelos também só conseguem distinguir carros com características que diferem de outros carros, por exemplo uma imagem de um HB20 Hatch ou Sedan na posição dianteira vão ser identificadas somente como HB20 por exemplo.

  O mesmo acontece com alguns modelos de Gol e Parati que tem frentes idênticas. Sendo assim certas diferenças entre veículos que não são visíveis olhando somente a imagem também não foram exploradas, como o ano do carro, alguns carros não tem variação de um ano para outro e versões como por exemplo câmbio manual ou automático, versões Confort, Plus, Turbo e etc foram características ignoradas.
  
	Só foram incluídos carros que tiveram mais de 10 imagens registradas em vídeos em locais públicos. Uma infinidade de modelos de  carro não foram encontrados e não foram buscados modelos específicos. Dado os objetivos do projeto isso não é um problema.
 

## Stack de Desenvolvimento

- Google Colab - Ambiente de Processamento - T4
- Google Drive - Armazenamento de Arquivos
- Python - Linguagem de programação
- CV2 - Processamento de Vídeos
- YOLOV8 - Detecção de carros e placas em imagens (frames de vídeo)
- TensorFlow + Keras - Biblioteca de Treinamento de Modelos de Inteligência Artificial
- EfficientNet - Modelos Pré-treinados
- SQLite3 - Banco de Dados

## Notebooks do Projeto

### 00_Criação Banco DSCARRO

Criação do arquivo de banco do SQLite3 e da tabela CARRO, nesta tabela que serão gravadas as informações do nome do vídeo, número do frame e localização do carro, placa e marca em um primeiro momento. Em um segundo momento serão inseridas as informações de posição do carro e modelo.

### 01_Identificador_Carros_Placas

Processamento do vídeo e gravação das informações encontradas frame a frame na tabela carro. 

Este notebook faz a identificação se existe um carro com placa, caso a distância entre o centro da placa e o centro do carro e está dentro dos limites de tolerância a informação do nome do vídeo, frame e das posições do carro, placa e marca são inseridas no banco de dados. Neste momento ainda não foi realizada nenhuma classificação. 

A posição da marca de carros na maioria das vezes está acima da placa. Tendo as coordenadas da placa a posição da marca foi calculada como para ficar abaixo das coordenadas Y do carro e acima das coordenadas Y da placa e tomada as coordenadas X da placa.

Com as marcas é possível criar um modelo YOLO de detecção. Para isto é necessário criar um banco de marcas com labels. Existem soluções gratuitas para realizar a criação do dataset de anotação de marcas.

Aqui são dependendo do setup de filmagem como por exemplo quantidade de frames por segundo, velocidade em que a câmera está em relação ao estacionamento, múltiplas imagens do mesmo veículo são registradas em diversos ângulos. 

O track da biblioteca YOLO mantém um ID para aquele carro ao longo do vídeo, assim um mesmo veículo com múltiplas imagens recebe o mesmo número de identificação do track. Essa informação foi utilizada mais pra frente, realizando a identificação do modelo de uma de múltplas imagens, a informação foi replicada via operações de banco de dados para todos os carros com o mesmo número de identificação de track.


### 02_Criacao_Pastas_Imgs_Unicas

Nesse passo foram criadas 3 grandes pastas. Imagens de carros, placas e marcas. Nelas foram gravadas imagens únicas de cada carro. 

Nos vídeos existem diversos frames com um mesmo modelo de placa, foi considerada a melhor imagem dentre várias de um mesmo carro, aquela cuja distância entre o centro do carro e da placa mais se aproxima de zero.

![melhor_imagem_banco_dados](https://github.com/user-attachments/assets/71c2d448-5ad3-4571-afa9-28445a13901b)




Realizando a consulta no banco que retorna a melhor imagem para cada carro em cada vídeo. A consulta retorna as informações de qual era o vídeo, frame e as posições do carro, placa e marca.
Com essas informações é possível com uma função abrir o vídeo, no frame exato e fazer um recorte (crop) da imagem e salvá-la em uma pasta. 
Nesta etapa o nome das imagens são salvas com  número único da ID_TABELA. Assim cada trio de imagens do mesmo carro tem o mesmo nome nas 3 pastas.

Ao final é criado um arquivo zip e feito o upload para o ambiente do google drive.

Etapa Manual

Nesse ponto a pasta com imagens de carros tem carros de todos os modelos e posições e misturados.
A primeira classificação em pastas de modelos e posição foi feita manualmente, dado que não existe modelo prévio de classificação.
Foram criadas pastas para cada classe (modelo) e inseridas as imagens nelas.


### 03_aplicacao_classe_imagens_banco

Após a etapa manual de classificação das imagens únicas esse notebook realiza a replicação da informação da classe e posição para todos os registros do mesmo vídeo e ID_CARRO_VIDEO


Antes
![antes](https://github.com/user-attachments/assets/e4d4ca56-df67-4bda-a10d-cb3f6e8fa8ce)


Depois
![depois](https://github.com/user-attachments/assets/fa475564-d087-4f28-91f1-a9ff509833ef)






### 04_Extracao_Todas_Imagens_Video.ipynb
Notebook que faz a extração de todas as imagens que estão classificadas no banco de dados.
Nesse ponto é gerada uma pasta com todas as imagens classificadas de todos os carros a partir de um vídeo.


### 05_Gerador_Dataset

Notebook que faz a geração do Dataset manipulando as imagens das pastas. Neste ponto não é feita mais manipulação de vídeo, somente de imagens já classificadas.

A geração do dataset inicia com a consulta no banco de dados. A partir da consulta um dataset é gerado. 

O dataset é composto de três pastas: treino, teste e validação. As imagens são inseridas nas pastas de maneira aleatória para evitar.

Com este notebook é possível gerar datasets com todos os modelos disponíveis ou com modelos específicos.

### 06_Treino_de_Modelo

Finalmente no último notebook acontece o treinamento do modelo utilizando o dataset criado no notebook anterior.

Aqui é realizado o download do dataset e do modelo pré-treinado da família do EfficientNet, realizado o setup de treinamento.

São criados os ImageGenerators necessários.

Camadas extras são adicionadas ao modelo base.

Realizado o Treinamento.

Pós treinamento são realizadas predições e análise dos resultados.

