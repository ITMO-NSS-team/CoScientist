import base64
from io import BytesIO
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from PIL import Image
from pydantic import BaseModel, Field


class ExpandedSummary(BaseModel):
    """
    Expanded version of paper's summary.
    """
    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description="Title of the paper. If the title is not explicitly specified, use the default value - 'NO TITLE'"
    )
    publication_year: int = Field(
        description=(
            "Year of publication of the paper. If the publication year is not explicitly specified, use the default "
            "value - 9999."
        )
    )
    authors: str = Field(
        description=(
            "Authors of the paper: a string of comma separated first and last names or surnames and initials. "
            "If the authors are not explicitly specified, use the default value 'NO AUTHORS'."
        )
    )
    source: str = Field(
        description=(
            "Source where the paper was published. If the source is not explicitly specified, use the default "
            "value - 'UNDEFINED'"
        )
    )
    research_area: str = Field(
        description=(
            "Area or field of science the paper is about. If the area is hard to determine, use the default value "
            "- 'OTHER'"
        )
    )


def convert_to_base64(file_path, s3_store):
    """
    Convert an image file to a Base64 encoded string.

    This method reads an image from the specified file path, encodes it as a JPEG image in memory, then converts it
    into a Base64 string representation.

    Args:
        file_path (str): The path to the image file.
        s3_store: S3-like store with papers files

    Returns:
        str: A Base64 encoded string representing the JPEG image.
    """
    if file_path.startswith("http://"):
        s3_key, bucket_name = extract_s3_bucket_and_key(file_path)
        pil_image = Image.open(BytesIO(s3_store.get_image_bytes_from_s3(s3_key, bucket_name)))
    else:
        pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    """
    Creates a structured message containing text and images for use in a conversational context.

    This method prepares the input data into a format suitable for presenting information in a multi-modal interface,
    by converting images to data URIs that can be directly embedded in a message and combining them with the provided
    text.

    Args:
        data (dict): A dictionary containing the message content:
            - "text" (str): The text content of the message;
            - "image" (list): A list of base64 encoded JPEG images to include in the message.

    Returns:
        HumanMessage: A HumanMessage object with a structured 'content' list.
            The 'content' list contains dictionaries representing each part of the message,
            with "type" keys indicating whether it's "text" or "image_url". Image URLs
            are formatted as data URIs.
    """
    text = data["text"]
    imgs = data["image"]
    content_parts = []
    
    for img in imgs:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{img}",
        }
        content_parts.append(image_part)
    
    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)
    
    return HumanMessage(content=content_parts)


def extract_s3_bucket_and_key(s3_url: str):
    """
    Extracts the file key in S3 storage and the bucket name from the full file path.

    Args:
        s3_url: The full path to the file in S3 storage.

    Returns:
        A tuple of S3 key and bucket name.
    """
    o = urlparse(s3_url)
    bucket, key = o.path.split('/', 2)[1:]
    return key, bucket


def pil_to_base64(image_object, img_format="JPEG"):
    """
    Converts a PIL Image object to a base64 encoded string.

    Args:
        image_object: The PIL Image object.
        img_format: The image format for saving (e.g., "JPEG", "PNG").

    Returns:
        A base64 encoded string.
    """
    buffered = BytesIO()
    image_object.save(buffered, format=img_format)
    img_bytes = buffered.getvalue()
    img_b64bytes = base64.b64encode(img_bytes)
    img_b64string = img_b64bytes.decode('utf-8')
    return img_b64string


# TODO: delete after testing
def generate_mock_images(count: int = 5):
    images = {}
    for i in range(count):
        img = Image.new('RGB', (100, 100), color=(i * 50 % 256, 0, 0))
        images[f"picture_{i + 1}.jpeg"] = img
    return images


# TODO: delete after testing
def generate_dummy_html(simple_id: str):
    return f"""<!DOCTYPE html>
<html lang="ru">
<body>
    <h1>Влияние методов очистки HTML на качество извлечения научного контента номер {simple_id}</h1>
    <p>Иванов И.И.<sup>1</sup>, Петрова А.С.<sup>2</sup>, Сидоров Н.В.<sup>3</sup></p>
    <p><sup>1</sup>Московский государственный университет, <sup>2</sup>Санкт-Петербургский политехнический университет, <sup>3</sup>Новосибирский государственный университет</p>
    
    <h1>Аннотация</h1>
    <p>В данной работе исследуются алгоритмы очистки HTML-разметки от посторонних элементов (реклама, навигация, скрипты) с целью повышения точности парсинга научных статей. Показано, что удаление стилей и скриптов, а также выделение основного контента позволяет улучшить извлечение текста и изображений. В качестве тестовых данных использованы пять изображений: picture_1.jpeg — picture_5.jpeg, иллюстрирующих этапы обработки и результаты.</p>
    <p>Особое внимание уделяется сохранению структуры документа: заголовков, абзацев, таблиц, списков и подписей к рисункам. Предложенная функция удаляет только служебные элементы, не затрагивая значимый контент.</p>
    
    <h1>1. Введение</h1>
    <p>Современные научные публикации часто представлены в формате HTML, содержащем множество вспомогательных элементов: навигационные панели, рекламные блоки, скрипты аналитики, стили оформления. Для автоматического анализа больших массивов статей необходимо выделить значимый контент: заголовки, текст, формулы, таблицы и иллюстрации. Очистка HTML от мусора — первый и критически важный этап любого парсингового пайплайна.</p>
    <p>В последние годы появилось несколько подходов: от простого удаления тегов <code>&lt;style&gt;</code> и <code>&lt;script&gt;</code> до использования машинного обучения для идентификации основного текста. Наша работа опирается на эвристические правила, адаптированные под типичную структуру научных статей в открытых архивах.</p>
    <p>
        <img src="picture_1.jpeg" alt="Диаграмма точности извлечения">
        <figcaption>Рис. 1. Сравнение точности извлечения текста до и после очистки на корпусе из 500 статей.</figcaption>
    </p>
    
    <h1>2. Методы очистки</h1>
    <p>Нами разработана функция, выполняющая следующие шаги:
        <ol>
            <li>Удаление тегов <code>&lt;style&gt;</code>, <code>&lt;script&gt;</code>, <code>&lt;nav&gt;</code>, <code>&lt;footer&gt;</code>, <code>&lt;aside&gt;</code> и их содержимого.</li>
            <li>Удаление атрибутов классов, стилей и идентификаторов у всех оставшихся элементов.</li>
            <li>Сохранение только блочных элементов, содержащих текст: <code>&lt;p&gt;</code>, <code>&lt;h1&gt;–&lt;h6&gt;</code>, <code>&lt;table&gt;</code>, <code>&lt;ul&gt;</code>, <code>&lt;ol&gt;</code>, <code>&lt;figure&gt;</code> и т.д.</li>
            <li>Особая обработка изображений: сохраняются теги <code>&lt;img&gt;</code> с атрибутами <code>src</code> и <code>alt</code>, при этом удаляются обёртки из рекламных контейнеров.</li>
        </ol>
    </p>
    <p>Для проверки корректности работы мы подготовили набор тестовых HTML-документов, включающих различные варианты вёрстки. Ниже представлена схема предлагаемого алгоритма.</p>
    <p>
        <img src="picture_2.jpeg" alt="Схема работы алгоритма">
        <figcaption>Рис. 2. Детальная блок-схема алгоритма очистки HTML.</figcaption>
    </p>
    <p>Дополнительно была реализована эвристика для восстановления подписей к рисункам: если <code>&lt;figcaption&gt;</code> отсутствует, но сразу после изображения идёт короткий параграф, он может быть интерпретирован как подпись. В текущей версии такие подписи сохраняются вместе с рисунком.</p>
    <p>
        <img src="picture_4.jpeg" alt="Пример восстановления подписи">
        <figcaption>Рис. 4. Пример автоматического связывания изображения и подписи.</figcaption>
    </p>
    
    <h1>3. Экспериментальные результаты</h1>
    <p>Тестирование проводилось на коллекции из 100 научных статей, загруженных из репозиториев arXiv и PubMed Central. Измерялась полнота сохранения текста и изображений, а также точность удаления мусорных элементов. После очистки все пять целевых изображений (picture_1.jpeg – picture_5.jpeg) были успешно извлечены в 98% случаев.</p>
    <p>
        <img src="picture_3.jpeg" alt="График зависимости полноты от порога удаления">
        <figcaption>Рис. 3. Зависимость полноты извлечения от агрессивности очистки (порога удаления элементов).</figcaption>
    </p>
    <p>В статье также присутствуют формулы, например: E = mc<sup>2</sup>, интегралы ∫ f(x) dx и химические соединения H<sub>2</sub>O. Для таблиц проверялась сохранность структуры строк и столбцов.</p>
    <table border="1">
        <caption>Таблица 1. Сравнение методов очистки по метрикам точности и полноты</caption>
        <thead>
            <tr><th>Метод</th><th>Точность (Precision)</th><th>Полнота (Recall)</th><th>F1-мера</th></tr>
        </thead>
        <tbody>
            <tr><td>Предложенный алгоритм</td><td>0.95</td><td>0.98</td><td>0.96</td></tr>
            <tr><td>Базовый (только удаление style/script)</td><td>0.82</td><td>0.79</td><td>0.80</td></tr>
            <tr><td>Эвристический (Boilerpipe)</td><td>0.88</td><td>0.91</td><td>0.89</td></tr>
        </tbody>
    </table>
    <p>Дополнительно была измерена скорость работы: в среднем 0.02 секунды на документ (Python + BeautifulSoup), что позволяет обрабатывать большие коллекции в реальном времени.</p>
    
    <h1>4. Обсуждение</h1>
    <p>Несмотря на высокие показатели, предложенный метод имеет ограничения. Например, если научная статья использует нестандартные теги (например, <code>&lt;div&gt;</code> для каждого абзаца), такие абзацы могут быть ошибочно удалены. В будущем планируется добавить адаптивные правила на основе машинного обучения.</p>
    <p>Ещё одна сложность — корректное извлечение изображений, встроенных в сложные вёрстки (галереи, карусели). В тестовом наборе таких случаев не было, но в реальных данных они встречаются. Для них может потребоваться анализ CSS-свойств.</p>
    <p>
        <img src="picture_5.jpeg" alt="Пример сложной вёрстки с изображением">
        <figcaption>Рис. 5. Тестовый случай с изображением внутри галереи.</figcaption>
    </p>
    
    <h1>5. Заключение</h1>
    <p>Разработанная функция очистки HTML показала высокую эффективность при парсинге научных статей. Все ключевые элементы, включая пять тестовых изображений, сохраняются, а мусорные блоки успешно удаляются. Код функции доступен в репозитории и может быть легко адаптирован под другие типы документов.</p>
    
    <h1>Благодарности</h1>
    <p>Работа выполнена при поддержке гранта РФФИ № 25-01-00123. Авторы благодарят коллег из Института вычислительных технологий за ценные замечания.</p>
    
    <h1>References</h1>
    <ol>
        <li>Smith J., Johnson L. Cleaning HTML for Text Mining: A Comprehensive Review. Journal of Data Science, 2025, vol. 12, no. 3, pp. 45–67.</li>
        <li>Петров В.В., Сидоров Н.В. Методы извлечения информации из веб-страниц на основе DOM-структуры. Программирование, 2024, № 2, с. 112–125.</li>
        <li>Kohlschütter C., Fankhauser P., Nejdl W. Boilerplate Detection using Shallow Text Features. In Proceedings of the Third ACM International Conference on Web Search and Data Mining (WSDM), 2010, pp. 441–450.</li>
        <li>Иванов И.И. Очистка HTML от рекламы с помощью регулярных выражений: за и против. Труды конференции "Анализ данных-2025", с. 78–82.</li>
    </ol>
</body>
</html>"""
