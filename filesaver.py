import numpy as np


def WriteHuffmanTable(f, huff_table, table_class, destination_id):
    f.write(b'\xFF\xC4')

    length = 3 + sum(len(huff_table[symbol]) for symbol in huff_table)
    f.write(length.to_bytes(2, 'big'))

    table_info = (table_class << 4) | destination_id
    f.write(bytes([table_info]))

    symbol_counts = [0] * 16
    for code in huff_table.values():
        symbol_counts[len(code) - 1] += 1
    
    f.write(bytes(symbol_counts))

    sorted_symbols = sorted(huff_table.keys(), key=lambda k: len(huff_table[k]))
    for symbol in sorted_symbols:
        f.write(bytes([symbol]))

    for symbol in sorted_symbols:
        code_length = len(huff_table[symbol])
        code_value = int(huff_table[symbol], 2)
        f.write(code_value.to_bytes((code_length + 7) // 8, 'big'))

def WriteQuantizationTable(f, quant_table, table_id):
    f.write(b'\xFF\xDB')
    f.write((67).to_bytes(2, 'big'))
    
    f.write((0 << 4 | table_id).to_bytes(1, 'big'))
    
    f.write(quant_table.tobytes())

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
        else:
            f.write((0x11).to_bytes(1, 'big'))
        f.write((0).to_bytes(1, 'big'))

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

def WriteJpeg(encoded_dc, encoded_ac, dc_huff_table, ac_huff_table, quant_table_luminance, quant_table_chrominance, image_width, image_height, addr):
    with open(addr, 'wb') as f:
        f.write(b'\xff\xd8')
        
        WriteQuantizationTable(f, quant_table_luminance, 0)
        WriteQuantizationTable(f, quant_table_chrominance, 1)

        WriteStartOfFrame(f, image_width, image_height, 3)

        WriteHuffmanTable(f, dc_huff_table, 0, 0)  # DC Huffman table for Y
        WriteHuffmanTable(f, ac_huff_table, 1, 0)  # AC Huffman table for Y
        WriteHuffmanTable(f, dc_huff_table, 0, 1)  # DC Huffman table for Cb/Cr
        WriteHuffmanTable(f, ac_huff_table, 1, 1)  # AC Huffman table for Cb/Cr
        
        WriteStartOfScan(f, 3)

        WriteCompressedData(f, encoded_dc, encoded_ac)

        f.write(b'\xff\xd9')