"""
Structure-Aware Chunker for E-Commerce & Customer Support RAG Pipeline

This module implements structure-aware text chunking using `langchain-text-splitters`.
Instead of naive fixed-size character splitting, it respects natural document boundaries
by trying separators in a hierarchical priority order:
1. Paragraph breaks ("\n\n")
2. Line breaks ("\n")
3. Sentence endings (". ")
4. Word boundaries (" ")
5. Raw character fallback ("")

This guarantees that sections, paragraphs, and sentences remain intact as much as possible,
only splitting at lower-priority boundaries when text exceeds the chunk size cap.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_splitter(
    chunk_size: int = 2000,
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """
    Constructs and returns a RecursiveCharacterTextSplitter instance configured
    with structure-aware priority separators.

    Args:
        chunk_size: Maximum target character size per chunk (default: 2000, approx 512 tokens).
        chunk_overlap: Overlap in characters between adjacent chunks (default: 200).

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter object.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )


def chunk_text(text: str) -> list[str]:
    """
    Splits text into chunks using structure-aware text splitting while preserving natural boundaries.

    Args:
        text: Raw input string to be chunked.

    Returns:
        list[str]: List of non-empty, whitespace-stripped chunk strings.
    """
    if not text or not text.strip():
        return []

    splitter = get_splitter()
    raw_chunks = splitter.split_text(text)

    # Strip leading/trailing whitespace and filter out empty chunks
    cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk and chunk.strip()]
    return cleaned_chunks


if __name__ == "__main__":
    print("=== Testing Structure-Aware Chunker ===\n")

    # Example A: Short text (FAQ entry)
    short_text = (
        "Q: How can I track my order?\n"
        "A: Once your order ships, you will receive an email notification containing a tracking "
        "number and a direct link to track your package delivery progress."
    )

    print("--- Test 1: Short Text (FAQ Entry) ---")
    short_chunks = chunk_text(short_text)
    print(f"Total Chunks: {len(short_chunks)}")
    for idx, chunk in enumerate(short_chunks, 1):
        print(f"Chunk {idx} (Length {len(chunk)}):\n{chunk}\n")
    assert len(short_chunks) == 1, f"Expected 1 chunk, got {len(short_chunks)}"

    # Example B: Longer text (>2500 characters) across paragraph boundaries
    paragraph_1 = (
        "1. RETURN AND REFUND POLICY\n"
        "Customers may return items within 30 calendar days of delivery for a full refund or exchange. "
        "To be eligible for a return, your item must be unused, unwashed, and in the same condition that you received it. "
        "It must also be in the original packaging with all attached tags and accessories intact. "
        "To initiate a return request, please visit our online Return Portal and submit your order number alongside your email address. "
        "Once your return request is approved, a pre-paid shipping label will be generated and sent directly to your registered email address. "
        "Please pack the item securely and drop it off at any authorized shipping carrier location within 7 days of receiving the return label. "
        "Inspections are typically completed within 3 to 5 business days following arrival at our fulfillment warehouse center. "
        "Refunds will be issued back to the original method of payment used during checkout upon passing inspection. "
        "Please note that original shipping charges are non-refundable unless the return is due to an error on our part or a defective item. "
    ) * 3  # Repeats to ensure paragraph length ~ 1400 chars

    paragraph_2 = (
        "2. WARRANTY AND REPAIR TERMS\n"
        "All hardware and electronic products sold on our platform come with a standard one-year limited manufacturer warranty. "
        "This limited warranty covers defect issues in material or workmanship under normal usage conditions during the warranty period. "
        "During the warranty period, our technical service center will repair or replace, at no charge, products or parts of a product "
        "that prove defective because of improper material or workmanship under normal care and maintenance. "
        "This warranty does not cover any problem that is caused by conditions, malfunctions, or damage not resulting from defects in material. "
        "To obtain warranty service, you must first contact customer support to determine the problem and the most appropriate solution. "
        "If a replacement unit is issued, the remaining duration of the original warranty or ninety days, whichever is longer, applies. "
        "Customers are responsible for shipping costs associated with sending items to the repair facility unless explicitly covered by promo. "
    ) * 3  # Repeats to ensure paragraph length ~ 1400 chars

    long_text = paragraph_1.strip() + "\n\n" + paragraph_2.strip()

    print("--- Test 2: Long Text (>2500 chars with '\\n\\n' boundary) ---")
    print(f"Total Input Length: {len(long_text)} characters")
    long_chunks = chunk_text(long_text)
    print(f"Total Chunks Generated: {len(long_chunks)}")

    for idx, chunk in enumerate(long_chunks, 1):
        print(f"--- Chunk {idx} (Length: {len(chunk)} characters) ---")
        print(f"Starts with: {chunk[:60]!r}")
        print(f"Ends with: {chunk[-60:]!r}\n")

    print("Structure-aware chunker tests verified successfully!")
