from bitarray import bitarray
import numpy as np
from collections import Counter, OrderedDict
from .huffman import HuffmanTable

def WriteAPP0(f):
    f.write(b'\xff\xe0')  

    length = 16
    f.write(length.to_bytes(2, 'big'))

    f.write(b'JFIF\x00')
    f.write(b'\x01\x01')
    f.write(b'\x00')
    f.write(b'\x00\x01\x00\x01')
    f.write(b'\x00\x00')

def WriteHuffmanTable(f, huff_table: OrderedDict, table_class, destination_id):
    f.write(b'\xFF\xC4')
    
    length = 3 + len(huff_table) + 16
    f.write(length.to_bytes(2, 'big'))
    
    table_info = (table_class << 4) | destination_id
    f.write(bytes([table_info]))

    f.write(CalHTBitLength(huff_table))
    
    for key, _ in huff_table.items(): 
        f.write(key.to_bytes(1, 'big'))

def WriteQuantizationTable(f, quant_table: np.ndarray, table_id):
    f.write(b'\xFF\xDB')

    f.write((67).to_bytes(2, 'big'))
    f.write((0 << 4 | table_id).to_bytes(1, 'big'))

    direction = 1
    rows, cols = quant_table.shape
    row, col = 0, 0
    for _ in range(quant_table.size):
        f.write(int(quant_table[row, col]).to_bytes(1, 'big'))
        if direction == 1:
            if col == cols - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            if row == rows - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1

def WriteStartOfFrame(f, image_width, image_height, num_components):
    f.write(b'\xFF\xC0')
    f.write((8 + 3 * num_components).to_bytes(2, 'big'))
    
    f.write((8).to_bytes(1, 'big'))
    f.write(image_height.to_bytes(2, 'big'))
    f.write(image_width.to_bytes(2, 'big'))
    f.write((num_components).to_bytes(1, 'big'))
    
    for i in range(1, num_components + 1):
        f.write((i).to_bytes(1, 'big'))
        if i == 1:
            f.write((0x11).to_bytes(1, 'big'))
            f.write((0).to_bytes(1, 'big'))
        else:
            f.write((0x11).to_bytes(1, 'big'))
            f.write((1).to_bytes(1, 'big'))
        
def WriteStartOfScan(f, num_components):
    f.write(b'\xFF\xDA')
    f.write((6 + 2 * num_components).to_bytes(2, 'big'))
    
    f.write((num_components).to_bytes(1, 'big'))
    
    for i in range(1, num_components + 1):
        f.write((i).to_bytes(1, 'big'))
        if i == 1:
            f.write((0 << 4 | 0).to_bytes(1, 'big'))
        else:
            f.write((1 << 4 | 1).to_bytes(1, 'big'))
            
    f.write((0).to_bytes(1, 'big'))
    f.write((63).to_bytes(1, 'big'))
    f.write((0).to_bytes(1, 'big'))

def WriteCompressedData(f, encoded_dc, encoded_ac):
    buffer = ''

    for dc, ac in zip(encoded_dc, encoded_ac):
        buffer += dc + ac

        while len(buffer) >= 8:
            byte = int(buffer[:8], 2)
            f.write(bytes([byte]))

            if byte == 0xFF:
                f.write(b'\x00')
            
            buffer = buffer[8:]

    if buffer:
        byte = int(buffer.ljust(8, '0'), 2)
        f.write(bytes([byte]))

        if byte == 0xFF:
            f.write(b'\x00')

def EncodeCoefficient(value):
    if value == 0:
        return 0, ''
    abs_value = abs(value)
    category = int(np.floor(np.log2(abs_value))) + 1
    if value < 0:
        bit_code = format((1 << category) - 1 + value, '0{}b'.format(category))
    else:
        bit_code = format(value, '0{}b'.format(category))
    return category, bit_code

def EncodeBlock(dc_value, ac_values, dc_huff_table, ac_huff_table):
    encoded_block = ''

    dc_category, dc_bits = EncodeCoefficient(dc_value)
    encoded_block += dc_huff_table[dc_category] + dc_bits

    zero_count = 0
    for ac in ac_values:
        if ac == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                encoded_block += ac_huff_table[0xF0]
                zero_count -= 16
            ac_category, ac_bits = EncodeCoefficient(ac)
            encoded_block += ac_huff_table[zero_count << 4 | ac_category] + ac_bits
            zero_count = 0
    
    if zero_count > 0:
        encoded_block += ac_huff_table[0x00]
    
    return encoded_block

def CalHTBitLength(huff_table: OrderedDict):
    length_counts = [0] * 16
    for _ , code in huff_table.items():
        length = len(code)

        if length <= 16:
            length_counts[length - 1] += 1
            
    record = bytearray(length_counts)
    bit_array = bitarray()
    bit_array.frombytes(record)
    
    return bit_array

def WriteJpeg(bitStream: bitarray, tables:HuffmanTable, quant_table_luminance, quant_table_chrominance, image_height, image_width, addr):
    with open(addr, 'wb') as f:
        f.write(b'\xff\xd8')

        # Check X,Y pixel density
        WriteAPP0(f)
        
        WriteQuantizationTable(f, quant_table_luminance, 0)
        WriteQuantizationTable(f, quant_table_chrominance, 1)

        WriteStartOfFrame(f, image_width, image_height, 3)

        WriteHuffmanTable(f, tables.dcLuminanceCodes, 0, 0)  # DC Huffman table for Y
        WriteHuffmanTable(f, tables.acLuminanceCodes, 1, 0)  # AC Huffman table for Y
        WriteHuffmanTable(f, tables.dcChrominanceCodes, 0, 1)  # DC Huffman table for Cb/Cr
        WriteHuffmanTable(f, tables.acChrominanceCodes, 1, 1)  # AC Huffman table for Cb/Cr
        
        WriteStartOfScan(f, 3)

        f.write(bitStream.tobytes())

        f.write(b'\xff\xd9')