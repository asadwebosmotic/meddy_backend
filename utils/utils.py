from pydantic import BaseModel
import base64
from pydantic import Field
from typing import ClassVar
from tempfile import NamedTemporaryFile

class PageText(BaseModel):
    page_number: int
    text: str
    filename: str

class PdfExtractionResult(BaseModel):
    pages: list[PageText]

class PdfExtractor:
    @staticmethod
    def _save_temp_file(content: bytes, suffix: str = ".pdf") -> str:
        """Save content to temporary file and return path"""
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            return tmp.name

    @staticmethod
    def _encode_file(file_location: str) -> str:
        """Encode the file to base64."""
        try:
            with open(file_location, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_location} was not found.")

class Image(BaseModel):
    image_description: ClassVar[str] = """
    Identify the exact type of medical visualization or image with precise terminology.
    Classify from: line chart, bar graph, pie chart, scatter plot, table, flowchart,
    organizational chart, infographic, photograph, diagram, or other (specify).
    Include the chart's orientation if applicable (vertical/horizontal/radial).
    """

    summary_description: ClassVar[str] = """
    Provide details about the medical data in the image, including:
    1. All quantitative data points (exact numbers, percentages, values)
    2. Clear description of axes (variables, units, scales)
    3. Key trends (growth/decline patterns, correlations, anomalies)
    4. Time periods covered (if temporal data)
    5. Comparisons being made (between products, time periods, categories)
    6. Notable visual elements (annotations, highlights, outliers)
    7. The core medical insight or conclusion conveyed
    8. Any text labels or legends that appear in the image
    9. Confidence level in interpretation (high/medium/low)

    Structure as: [Data Summary] -> [Trend Analysis] -> [Key Takeaways]
    """

    image_type: str = Field(..., description=image_description)
    summary: str = Field(..., description=summary_description)
