import torch
import torch.nn.functional as F

from config import Config
from model import CharLSTM
from text_utils import CharTokenizer


def generate_text(model, tokenizer, start_text, length, device, temperature=1.0):
    """
    使用训练好的字符RNN生成文本

    参数
    ----------
    model : CharRNN
        训练好的模型

    tokenizer : CharTokenizer
        字符tokenizer

    start_text : str
        生成的起始prompt

    length : int
        需要生成的字符数量

    device : torch.device

    temperature : float
        控制随机程度
    """

    model.eval()

    # ------------------------------------------------
    # Step1 把prompt转为token id
    # ------------------------------------------------

    ids = tokenizer.encode(start_text)

    # 例如:
    # "To be"
    #
    # ids:
    # [45,12,8,8,3]

    result = start_text

    for _ in range(length):

        # ---------------------------------------------
        # Step2 构造输入tensor
        # ---------------------------------------------

        input_tensor = torch.tensor(ids).unsqueeze(0).to(device)

        # shape
        #
        # (1, seq_len)
        #
        # 1 = batch

        with torch.no_grad():

            logits, _ = model(input_tensor)

        # logits shape
        #
        # (1, seq_len, vocab_size)

        # ---------------------------------------------
        # Step3 取最后一个time step预测
        # ---------------------------------------------

        last_logits = logits[:, -1, :]

        # shape
        #
        # (1, vocab_size)

        # ---------------------------------------------
        # Step4 temperature调整
        # ---------------------------------------------

        last_logits = last_logits / temperature

        # softmax得到概率
        probs = F.softmax(last_logits, dim=-1)

        # shape
        #
        # (1, vocab_size)

        # ---------------------------------------------
        # Step5 按概率采样
        # ---------------------------------------------

        next_id = torch.multinomial(probs, num_samples=1)

        # shape
        #
        # (1,1)

        next_id = next_id.item()

        # ---------------------------------------------
        # Step6 转回字符
        # ---------------------------------------------

        next_char = tokenizer.decode([next_id])

        result += next_char

        # 更新输入序列
        ids.append(next_id)

    return result


def main():

    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------
    # Step1 读取数据
    # ------------------------------------------------

    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text[:60000]

    tokenizer = CharTokenizer(text)

    # ------------------------------------------------
    # Step2 构建模型
    # ------------------------------------------------

    model = CharLSTM(
        tokenizer.vocab_size,
        config.embedding_dim,
        config.hidden_size,
        config.num_layers
    ).to(device)

    # ------------------------------------------------
    # Step3 加载训练好的参数
    # ------------------------------------------------

    model.load_state_dict(
        torch.load("./saved_models/temp_params.pt", map_location=device)
    )
    print("Model loaded.")

    # ------------------------------------------------
    # Step4 生成文本
    # ------------------------------------------------

    prompt = "To be"

    generated = generate_text(
        model,
        tokenizer,
        prompt,
        length=300,
        device=device,
        temperature=0.8
    )

    print("\nGenerated text:\n")
    print(generated)


if __name__ == "__main__":
    main()