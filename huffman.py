class TreeNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

class HuffmanTree:
    def __init__(self, text):
        self.text = text
        self.freq_map = self.build_freq_map()
        self.root = self.build_huffman_tree()
        self.codes = self.generate_huffman_codes()

    def build_freq_map(self):
        freq_map = {}
        for char in self.text:
            freq_map[char] = freq_map.get(char, 0) + 1
        return freq_map

    def build_huffman_tree(self):
        nodes = [TreeNode(char, freq) for char, freq in self.freq_map.items()]
        while len(nodes) > 1:
            nodes.sort(key=lambda x: x.freq)
            left = nodes.pop(0)
            right = nodes.pop(0)
            merged = TreeNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            nodes.append(merged)
        return nodes[0]

    def generate_huffman_codes(self):
        codes = {}
        def traverse(node, code=''):
            if node:
                if node.char:
                    codes[node.char] = code
                traverse(node.left, code + '0')
                traverse(node.right, code + '1')
        traverse(self.root)
        return codes

    def encode(self, original_text):
        encoded_text = ''
        for char in original_text:
            encoded_text += self.codes[char]
        return encoded_text

    def decode(self, encoded_text):
        decoded_text = ''
        current_node = self.root
        for bit in encoded_text:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            if not current_node.left and not current_node.right:
                decoded_text += current_node.char
                current_node = self.root
        return decoded_text