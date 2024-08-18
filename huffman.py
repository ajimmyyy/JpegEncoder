import heapq
from collections import defaultdict, Counter
import numpy as np

class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def BuildHuffmanTree(freqs):
    heap = [Node(symbol, freq) for symbol, freq in freqs.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def GenerateHuffmanCodes(node, prefix="", huff_table={}):
    if node is not None:
        if node.symbol is not None:
            huff_table[node.symbol] = prefix
        GenerateHuffmanCodes(node.left, prefix + "0", huff_table)
        GenerateHuffmanCodes(node.right, prefix + "1", huff_table)
    return huff_table

def EncodeValue(value):
    abs_value = abs(value)
    category = int(np.floor(np.log2(abs_value))) + 1 if abs_value != 0 else 0
    bit_string = format(abs_value, f'0{category}b')

    if value < 0:
        bit_string = ''.join('1' if bit == '0' else '0' for bit in bit_string)
    return category, bit_string

def GenerateHuffmanTable(data: np.ndarray):
    freq_counter = defaultdict(int)
    
    for value in data:
        category, _ = EncodeValue(value)
        freq_counter[category] += 1
    
    huffman_tree = BuildHuffmanTree(freq_counter)
    huff_table = GenerateHuffmanCodes(huffman_tree)

    return huff_table

def HuffmanEncode(data, huff_table):
    encoded_data = ""
    for value in data:
        category, bit_string = EncodeValue(value)
        encoded_data += huff_table[category] + bit_string
    return encoded_data

def EncodeDCAC(dc_matrix, ac_matrix):
    dc_data = dc_matrix.flatten()
    ac_data = ac_matrix.flatten()
    
    dc_huff_table = GenerateHuffmanTable(dc_data)
    ac_huff_table = GenerateHuffmanTable(ac_data)

    encoded_dc = HuffmanEncode(dc_data, dc_huff_table)
    encoded_ac = HuffmanEncode(ac_data, ac_huff_table)

    return encoded_dc, encoded_ac, dc_huff_table, ac_huff_table