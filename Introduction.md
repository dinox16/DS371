# Mục tiêu
Xây dựng hệ thống nhận diện cử chỉ tay để điều khiển PowerPoint thông qua webcam:
- Nhận diện chính xác
- Hoạt động real-time
- Tương tác mượt

# Dataset

Sử dụng **20BN-JESTER**:
- Video cử chỉ tay
- Nhiều lớp (swipe left, right, ...)
- Dữ liệu lớn

# Phương pháp

## Trích xuất đặc trưng

Sử dụng MediaPipe để lấy landmark bàn tay:

$$
X = \{(x_i, y_i, z_i)\}_{i=1}^{N}
$$

## Biểu diễn dữ liệu

Chuỗi frame được biểu diễn:

$$
X \in \mathbb{R}^{T \times N \times 3}
$$

Trong đó:
- $T$: số frame
- $N$: số landmark

# Ứng dụng

- Swipe right → Next slide  
- Swipe left → Previous slide  
- Fist → Play/Pause

# Nội dung lý thuyết
- Xây dựng "từ điển" cho LSTM (Long-short term memory)

Biến đổi dữ liệu từ hình học sang dữ liệu ngôn ngữ (Tokenization)
 Tức là, mỗi landmarks cho chúng ta một vector như sau 
 
 $$
 v = \begin{bmatrix} x \\ y \\ z \\ \end{bmatrix}
 $$ 
 
Từ đây ta suy ra được rằng một frame Mediapipe sẽ trả về một ma trận

$$ 
X = \begin{bmatrix}v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} 
$$

 Từ đây ta suy ra được rằng số chiều của ma trận này là một ma trận có số chiều là 
 $ 
 X \in \mathbb{R}^{63} 
 $
 
 Tiếp theo, chúng ta sẽ xây dựng một bộ từ điển cho mô hình LSTM. Với $k = 32$ với mỗi vector đặc trưng $v_i$ của một tư thế tay, thuật toán tính khoảng cách Euclid đến tất cả các tâm cụm: 
 
 $$ 
 d(x_i, \mu_j) = \sqrt{\sum_{d=1}^{63} (x_{id} - \mu_{jd})^2} 
 $$
 
 Mỗi tư thế tay sẽ được gán vào cụm có khoảng cách ngắn nhất.
 Sau khi gán xong, mỗi tâm cụm $\mu_j$ được tính toán lại bằng cách lấy trung bình cộng của tất cả các vector đã được gán cho nó 
 
 $$
 \mu_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i
 $$ 
 
 quá trình sẽ được lặp lại cho đến khi mô hình đã hội tụ. Kết quả mong muốn cuối cùng là thu được 32 "tư thế mẫu" đại diện cho mọi tư thế tay có thể xảy ra.
 Về bản chất, K-means thực hiện nhiệm vụ: Rời rạc hóa không gian đặc trưng liên tục. Có vẻ hơi khó hiểu nhưng thực sự thì sau khi qua K-means, ma trận 
 
 $$
 X = \begin{bmatrix}v_1 \\ v_2 \ \vdots \ v_n \end{bmatrix}  \xrightarrow{f} Y = \begin{bmatrix}0 \\ 1 \\  \vdots \\ 32 \end{bmatrix}
 $$ 
 
sẽ được gán một ID duy nhất, ví dụ như được gán với số 5 lúc này nó đóng vai trò là một "từ vựng" (token) đại diện cho một tư thế tay cụ thể (ví dụ: có thể là tư thế xòe tay). 
 
 Lý do lớn nhất mà chúng ta chọn việc này nằm ở chỗ LSTM học tốt nhất với dữ liệu thời gian, và thông thường được ứng dụng vào những bài toán giải quyết NLP, chúng ta tạo ra một bộ từ điển giống như bạn học thêm 1 thứ tiếng nào đó thì ít nhất phải có được một bộ từ điển về nó chứ không thể nào tự nhận biết được mà không cần sự hỗ trợ, ở đây chúng ta định nghĩa từ điển cho mô hình. Nhờ đó, việc nhận diện hành động trở nên trông giống việc máy tính đang đọc một đoạn văn bản chuyển động, giúp hệ thống chạy nhanh và ổn định. 
 
 Điều này cũng không phải là hoàn toàn tốt nhất, vẫn có những nhược điểm tiềm ẩn như nhảy token (xảy ra khi token đầu vào nhảy liên tục [5,6,5,6,5]), hay mẫu huấn luyện K-means không được chất lượng.

 - Huấn luyện LSTM

Chúng ta sẽ lấy 16 frame để dự đoán hành động, thì input lúc này 

$$
X_{\text{raw}} =
\begin{bmatrix}
[x_0, y_0, z_0, \ldots, z_{20}]_1 \\
[x_0, y_0, z_0, \ldots, z_{20}]_2 \\
\vdots \\
[x_0, y_0, z_0, \ldots, z_{20}]_{16}
\end{bmatrix} \xrightarrow{f} X_{\text{tokenized}} = [5, 1, 1, 2, 8, 8, 12, 12, 15, 15, 20, 20, 5, 5, 32, 32] 
$$

Nhìn vào đây ta có thể thấy rằng lúc này hành động diễn ra liên tục đã được làm giống như một đoạn văn. Giống với việc bạn Tokenizer khi sử dụng LSTM huấn luyện với bài toán NLP.

Output mong muốn đầu ra sẽ là nhãn của hành động. Ứng dụng của mô hình many-to-one.

Để có thể hiểu nhanh bản chất bạn có thể tham khảo ví dụ sau: Khi bạn recomment một bộ phim, bạn có thể miêu tả cảm giác của bản thân về bộ phim theo nhiều cách khác nhau nhưng chung quy lại vẫn là yêu thích hoặc không yêu thích hoặc là cảm thấy bình thường,... Tương tự ở đây, 16 frame được số hóa thành từng từ ngữ chung quy lại nó cũng chỉ miêu tả một hành động duy nhất.

Cuối cùng thì, LSTM chỉ cần học được ý nghĩa chuyển động, còn người định nghĩa cho chuyển động đó là K-means.

- Nhược điểm của hệ thống: 

Thiếu tính phân cấp : Việc duỗi thẳng 21 landmarks khiến mô hình coi rằng 21 điểm này là ngang hàng nhau, không phân biệt được cổ tay, lòng bàn tay hay ngón tay.

Lời nguyền đa chiều : Việc ma trận có số chiều quá lớn dễ khiến K-means bị nhầm lẫn về khoảng cách giữa chúng, không thể phản ánh đúng về mặt thị giác giữa chúng
