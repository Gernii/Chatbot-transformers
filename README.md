# Project: Chatbot - Transformers

## Thông tin về project:
* Sử dụng mô hình của transformers để tạo ra 1 Chatbot có thể xử lí dữ liệu nhanh và tốt.

## Packages
* Tensorflow >= 2.

## Dữ liệu

* Các cuộc phỏng vấn về tennis.
* Nguồn dữ liệu: [Tennis Corpus](http://zissou.infosci.cornell.edu/convokit/datasets/tennis-corpus/tennis-corpus.zip)

## TEFPA:
* T: input (1 câu hội thoại) output (1 câu hội thoại)
* E: $X, Y -> \widehat{Y}$
* F: 
* P: Sparse Categorical Cross Entropy
* A: optimizer: adam; learning rate costum, metrics : accuracy


## Cài đặt
- Tải dữ liệu về:
```
!git clone https://github.com/litahung/Chatbot-transformers.git
```
- Di chuyển đến vị trí chứa dữ liệu:
```
cd `đường dẫn`
```
- cài đặt các thư viện cần thiết:
```
!pip install -r requirements.txt
```
```
import gradio as gr
import main
```
- Hyperparameter:

```
main.Get_Hyper()
```
```
Kích thước 1 câu - max_length: 20
Batch size - batch_size: 256
Số lớp "encoder layer" và "decoder layer" - num_layers: 2
số lượng Units - num_units: 512
Số chiều Model - d_model: 256
Số lượng head trong Multi-head attention - num_heads: 8
Dropout - dropout: 0.1
Activation - activation: relu
Epochs - epochs: 20
Nếu muốn đổi thông tin Hyperparameter, vui lòng mở file main.py và đổi thông tin trong class Args rồi restart runtime
```
- Sử dụng:

```
iface = gr.Interface(fn=main.predict, inputs='text' , outputs="text")
iface.launch()
```

- Train mô hình:

```
main.train_model()
```


## Ưu điểm:
### xử lý đồng thời các từ thay vì xử lý tuần tự từng từ như mô hình Seq2Seq

![](https://i.imgur.com/W62Aqek.jpg)

### Self Attention Layer: 
* Cho phép mô hình mã hóa 1 từ có thể sử dụng thông tin của những từ liên quan tới nó (những từ đằng trước)

* Ví dụ: khi từ `nó` được mã hóa, nó sẽ chú ý vào các từ liên quan như là`mặt trời`.

![](https://pbcquoc.github.io/images/transformer/self_attention.jpg)

Cụ thể hơn, quá trình tính toán attention vector có thể được tóm tắt làm 3 bước như sau:

* Bước 1: Tính ma trận query, key, value bằng cách khởi tạo 3 ma trận trọng số query, key, vector. Sau đó nhân input với các ma trận trọng số này để tạo thành 3 ma trận tương ứng.
* Bước 2: Tính attention weights. Nhân 2 ma trận key, query vừa được tính ở trên với nhau để với ý nghĩa là so sánh giữa câu query và key để học mối tương quan. Sau đó thì chuẩn hóa về đoạn [0-1] bằng hàm softmax. 1 có nghĩa là câu query giống với key, 0 có nghĩa là không giống.
* Bước 3: Tính output. Nhân attention weights với ma trận value. Điều này có nghĩa là chúng ta biểu diễn một từ bằng trung bình có trọng số (attention weights) của ma trận value.
![](https://pbcquoc.github.io/images/transformer/attention_vector.jpg)

### Multi Head Attention

 Với mỗi self-attention, chúng ta học được một kiểu pattern, do đó để có thể mở rộng khả năng này, chúng ta đơn giản là thêm nhiều self-attention. Tức là chúng ta cần nhiều ma trận query, key, value mà thôi. Giờ đây ma trận trọng số key, query, value sẽ có thêm 1 chiều depth nữa.
 Multi head attention cho phép mô hình chú ý đến đồng thời những pattern dễ quan sát được như sau.

 * Chú ý đến từ kế trước của một từ
* Chú ý đến từ kế sau của một từ
* Chú ý đến những từ liên quan của một từ

![](https://pbcquoc.github.io/images/transformer/multi_head_attention.jpg)

## Tổng quan về mô hình Transformers cho Chatbot

![](https://raw.githubusercontent.com/bryanlimy/tf2-transformer-chatbot/master/transformer.png)

### Encoder

Gồm các phần chính sau:
* Input Embedding
* Positional Encoding
* `num_layers` encoder layers

Input của chúng ta được nhúng vào Embedding và kết hợp với Positional encoding để tạo thêm dữ liệu giúp mô hình nhận biết được vị trí của các từ trong câu nhờ Positional encoding. Sau đó ta đưa vào các encoder layers. Và cuối dùng đưa qua Decoder.

![](https://i.imgur.com/PeZhBRF.png)

#### Encoder layer

gồm các phần chính sau: 
* Multi-head attention (có padding mask)
* 2 lớp dense

Đầu ra của mỗi sublayer encoder layers là LayerNorm (x + Sublayer (x)). Quá trình chuẩn hóa được thực hiện trên chiều d_model cuối cùng.

![](https://i.imgur.com/7Y24kU3.png)


### Decoder
Gồm các phần chính sau:
* Output Embedding
* Positional Encoding
* `num_layers` decoder layers

Mục tiêu của chúng ta là nhúng vào Embedding và kết hợp với Positional encoding để tạo thêm dữ liệu giúp mô hình nhận biết được vị trí của các từ trong câu nhờ Positional encoding. Sau đó ta đưa vào các decoder layers để dự đoán. Và cuối dùng đưa qua 1 lớp Linear.
![](https://i.imgur.com/i2gYJQm.png)


#### Decoder layer
Gồm các phần chính sau:
* Masked multi-head attention (bao gồm look ahead mask và padding mask)
* Multi-head attention (bao gồm padding mask). value và key từ bộ dữ liệu output của encoder làm input. dữ liệu query từ output của masked multi-head attention.
* 2 lớp dense

decoder  dự đoán từ tiếp theo bằng cách tập trung vào output của encoder và  self-attention vào output của chính nó (output decoder).

![](https://i.imgur.com/CkVez18.png)

# Nguồn tài liệu:
> [Quoc Pham](https://pbcquoc.github.io/transformer/)
> 



