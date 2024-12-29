import bz2
import json
import os
from datetime import datetime

from loguru import logger
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")

def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])
                    
                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.
    
    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)
        
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    
    return queries, ground_truths, predictions

##################################################################
# if __name__ == "__main__":
#     from models.user_config import UserModel  # 載入用戶模型

#     # 初始化參與者模型
#     participant_model = UserModel()

#     # 單一測試問題
#     query = "is microsoft office 2019 available in a greater number of languages than microsoft office 2013?"

#     # 構建符合 batch_generate_answer 函數需求的字典
#     batch = {
#         "interaction_id": ["test_interaction_id"],  # 這裡使用一個範例 interaction_id
#         "query": [query],  # 將 query 包裝成列表
#         "query_time": ["2024-12-17T12:00:00"]  # 假設的查詢時間
#     }

#     # 模擬回答生成
#     print("正在生成預測結果...")
#     prediction = participant_model.batch_generate_answer(batch)

#     # 輸出生成結果
#     print(f"問題: {query}")
#     print(f"預測結果: {prediction}")
#############################################################################

if __name__ == "__main__":
    from models.user_config import UserModel  # 載入用戶模型

    # 設定資料集路徑
    DATASET_PATH = "example_data/dev_data.jsonl.bz2"

    # 1. 初始化參與者模型
    participant_model = UserModel()

    # 2. 生成預測結果
    print("正在生成預測結果...")
    queries, ground_truths, predictions = generate_predictions(DATASET_PATH, participant_model)

    # 3. 輸出生成結果
    output_file = "generated_predictions.json"
    output_data = [
        {"query": q, "prediction": p, "ground_truth": g}
        for q, p, g in zip(queries, predictions, ground_truths)
    ]

    # 將生成結果儲存到 JSON 檔案
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"預測結果已儲存至 {output_file}")
