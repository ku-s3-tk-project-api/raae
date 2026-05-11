import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

# Qiskit Core
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector

# Simulation & Math
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace

def normalize(input_vector):
  return ( 1/np.linalg.norm(input_vector) )* input_vector


def g(i) :  # binary grey code return functions
  return i ^ (i >> 1)


def ind(k) : # gives the control index required while appling the required c-x gates

  n = 2**k
  code = [g(i) for i in range(n)]

  control = []

  for i in range(n-1) :
    control.append(int(np.log2(code[i]^code[i+1])))
  control.append(int(np.log2(code[n-1]^code[0])))
  return control

def measure_selected_qubits(qc, qubit_indices):
    """
    Return a new circuit with measurements added on selected qubits.
    The original circuit remains unchanged.
    """

    # Create independent copy
    new_qc = qc.copy()

    num_measured = len(qubit_indices)
    c_reg = ClassicalRegister(num_measured, 'c_selected')
    new_qc.add_register(c_reg)

    for i, q_idx in enumerate(qubit_indices):
        new_qc.measure(q_idx, c_reg[i])

    return new_qc
        
class PAEDataEncoder:
    def __init__(self, parity_type="AAP"):
        self.parity_type = parity_type

    # ============================================================
    # PUBLIC METHOD
    # ============================================================
    def get_parameters(self, data, split_level):
        """
        Convert normalized classical vector into final θ parameters
        ordered exactly as required by BDSPPAECircuitBuilder.

        Output size = len(data) - 1
        """

        data = np.asarray(data, dtype=float)
        norm = np.linalg.norm(data)
        if norm == 0:
            raise ValueError("Input vector must not be zero.")
        data = data / norm

        n = int(np.log2(len(data)))
        if 2**n != len(data):
            raise ValueError("Length of data must be power of 2.")

        s = split_level
        num_blocks = 2**s
        block_size = len(data) // num_blocks
        q = n - s

        thetas = []

        # ========================================================
        # 1️⃣ TOP TREE (DC α's kept directly)
        # ========================================================
        full_alpha_levels = self._get_alpha_y_hierarchy(data)

        for level in range(s):
            thetas.extend(full_alpha_levels[level])

        # ========================================================
        # 2️⃣ BLOCKS
        # ========================================================
        blocks = [
            data[i*block_size:(i+1)*block_size]
            for i in range(num_blocks)
        ]

        for i, block in enumerate(blocks):

            block_alpha_levels = self._get_alpha_y_hierarchy(block)

            if i < num_blocks - 1:
                # ------------------------------
                # MOTTONEN BLOCK
                # Convert α → θ via M-matrix
                # ------------------------------
                for k, level_alphas in enumerate(block_alpha_levels):
                    theta_k = self._alpha_to_theta(level_alphas)
                    thetas.extend(theta_k)
            else:
                # ------------------------------
                # FINAL DC BLOCK
                # Keep α directly
                # ------------------------------
                for level_alphas in block_alpha_levels:
                    thetas.extend(level_alphas)

        return thetas

    # ============================================================
    # α HIERARCHY (Divide & Conquer)
    # ============================================================
    def _get_alpha_y_hierarchy(self, vec):
        """
        Returns list of α levels:
        [
          [α_root],
          [α_lvl1_0, α_lvl1_1],
          [α_lvl2_0, α_lvl2_1, ...],
          ...
        ]
        """

        vec = np.asarray(vec, dtype=float)
        n = int(np.log2(len(vec)))

        levels = []
        current = vec.copy()

        for level in range(n):
            next_level = []
            alphas = []

            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i+1]

                r = np.sqrt(left**2 + right**2)

                if r == 0:
                    alpha = 0.0
                else:
                    alpha = 2 * np.arcsin(right / r)

                alphas.append(alpha)
                next_level.append(r)

            levels.append(alphas)
            current = np.array(next_level)

        return levels[::-1]

    # ============================================================
    # α → θ (Mottonen M-matrix)
    # ============================================================
    def _alpha_to_theta(self, alphas):
        """
        Apply Mottonen M-matrix transform.
        Automatically detects k from the length of alphas.
        """
        alphas_arr = np.array(alphas)
        n = len(alphas_arr)

        # Calculate k: n = 2^k -> k = log2(n)
        k = int(np.log2(n))

        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # i ^ (i >> 1) is the Gray Code of i
                matrix[i, j] = (
                    2**(-k)
                    * (-1)**(bin(j & (i ^ (i >> 1))).count("1"))
                )

        return list(matrix @ alphas_arr)

def contraction_state(input_vector_):
    """Contract an N-dimensional state to N/2-dimensional state."""
    length = len(input_vector_)
    temp = input_vector_ ** 2
    result = np.zeros(length // 2)
    for i in range(length // 2):
        result[i] = (temp[2 * i] + temp[2 * i + 1]) ** 0.5
    return result


class HierarchicalEncoderRunner:
    """
    Runs the hierarchical encoding procedure over an input vector.

    Assumes:
      - encoder has method _get_alpha_y_hierarchy(vector) -> list of alphas
      - normalize(vector) is available in the current module
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def run(self, input_vector_):
        """
        Main entry: takes input_vector, computes dc/mot lengths,
        builds and returns result list.
        """
        N = len(input_vector_)
        level = int(np.log2(N))

        # Precompute section lengths
        self.dc_section_length = int(np.ceil(level / 2))
        self.mot_section_length = int(np.floor(level / 2))

        # Initialize result structure
        result = [[] for _ in range(level)]

        # Initial alphas for DC section
        alphas = self.encoder._get_alpha_y_hierarchy(input_vector_)
        for i in range(self.dc_section_length):
            result[i] = alphas[i]

        # Run recursive hierarchical filling
        self._isi_bersusun(
            start_level=level,
            input_vector_=input_vector_,
            result=result,
        )

        return result

    def flat_run(self, input_vector_):
        """
        Main entry: takes input_vector, computes dc/mot lengths,
        builds and returns result list.

        Then, flatten the output.
        """
        return flatten_list(self.run(input_vector_))

    def _isi_bersusun(self, start_level, input_vector_, result):
        """
        Recursive helper that fills 'result' depending on level parity.
        """
        level = int(np.log2(len(input_vector_)))
        if level <= 1:
            return

        #print("ini di level", level)

        if level % 2 == 0:
            self._handle_even_level(start_level, input_vector_, result)
        else:
            self._handle_odd_level(start_level, input_vector_, result)

    def _handle_even_level(self, start_level, input_vector_, result):
        """
        Even level branch (genap).
        """
        alphas = self.encoder._get_alpha_y_hierarchy(input_vector_)
        theta = self.encoder._alpha_to_theta(alphas[-1])

        # Overwrite slot for this level (start_level-1 is index)
        result[start_level - 1] = [[theta]]

        # Contract and recurse
        contracted = contraction_state(input_vector_)
        #print("input_vector", contracted)
        self._isi_bersusun(start_level - 1, contracted, result)

    def _handle_odd_level(self, start_level, input_vector_, result):
        """
        Odd level branch (ganjil).
        """

        half = len(input_vector_) // 2
        left_vec = normalize(input_vector_[:half])
        right_vec = normalize(input_vector_[half:])

        # Left side
        alphas_left = self.encoder._get_alpha_y_hierarchy(left_vec)
        theta_left = self.encoder._alpha_to_theta(alphas_left[-1])
        result[start_level - 1].extend([[theta_left]])

        # Right side
        alphas_right = self.encoder._get_alpha_y_hierarchy(right_vec)
        theta_right = self.encoder._alpha_to_theta(alphas_right[-1])
        result[start_level - 1].extend([[theta_right]])

        # Recurse on contracted halves
        left_contracted = contraction_state(left_vec)
        self._isi_bersusun(start_level - 1, left_contracted, result)

        right_contracted = contraction_state(right_vec)
        self._isi_bersusun(start_level - 1, right_contracted, result)

class StateConverter:
    def __init__(self):
        # Initialize the encoder for the flat_run capability
        self.encoder = HierarchicalEncoderRunner(PAEDataEncoder())

    def convert_to_params(self, input_vector):
        """Shortcut for flat_run to get circuit parameters."""
        return self.encoder.flat_run(input_vector)

    @staticmethod
    def to_probabilities(counts):
        """Infers qubit count and converts counts to a probability vector."""
        if not counts:
            return np.array([])
            
        n_qubits = len(next(iter(counts)))
        shots = sum(counts.values())
        
        probs = np.zeros(2**n_qubits)
        indices = [int(b[::-1], 2) for b in counts.keys()]
        values = [c / shots for c in counts.values()]
        
        probs[indices] = values
        return probs

    @staticmethod
    def to_amplitudes(counts):
        """Converts counts to an approximate amplitude vector."""
        return np.sqrt(StateConverter.to_probabilities(counts))

# Utility function: recursively add n to all integers in nested lists
add_number = lambda data, n: [
    add_number(item, n) if isinstance(item, list)
    else (item + n if isinstance(item, int) else item)
    for item in data
]


@dataclass
class RAAE_StructureResult:
    """Holds all the generated structure data."""
    structure_list: list
    last_number: int
    active_nodes: list
    node_types: list
    mottonen_cx_connections: list
    dc_cswap_connections: list

    def describe(self):
        """Nicely print structure details."""
        print(f"Last number (last_number): {self.last_number}")
        print(f"Active nodes (active_nodes): {self.active_nodes}")
        print("\nStructure layers (structure_list):")
        for i, struct in enumerate(self.structure_list):
            print(f"  Layer {i}: {struct}")
        print("\nNode types per layer (node_types):")
        for t in self.node_types:
            print(f"  {t}")
        # Uncomment to inspect connections
        print("\nMöttönen CX connections (mottonen_cx_connections):", self.mottonen_cx_connections)
        print("DC CSWAP connections (dc_cswap_connections):", self.dc_cswap_connections)


class RAAE_StructureBuilder:
    """Class that builds hierarchical quantum-like structures."""

    def __init__(self):
        pass  # Placeholder: could store configs if needed later

    # If the target output qubit is n
    # then level is n-1
    def build(self, level, start_num=0):
        """Build structure recursively up to a given level."""

        # Base case
        if level <= 0:
            return RAAE_StructureResult(
                structure_list=[[0]],
                last_number=0,
                active_nodes=[0],
                node_types=[['d']],
                mottonen_cx_connections=[[]],
                dc_cswap_connections=[[]]
            )

        # Odd level -> apply Möttönen-style extension
        if level % 2 == 1:
            prev = self.build(level - 1)
            prev_layer_size = len(prev.structure_list[-1])

            # Extend structure with new layer
            new_layer = [[] for _ in range(prev_layer_size)]
            new_layer.extend([prev.last_number + i + 1 for i in range(2 ** level)])
            prev.structure_list.append(new_layer)

            # Add node types ('m' for Möttönen)
            new_node_types = [[] for _ in range(prev_layer_size)]
            new_node_types.extend(['m' for _ in range(2 ** level)])
            prev.node_types.append(new_node_types)

            # Möttönen CX connections (using control indices)
            new_mottonen = [[] for _ in range(prev_layer_size)]
            target_qubit = len(prev.structure_list) - 1
            num_controls = len(prev.active_nodes)
            control_indices = ind(num_controls)  # Assumes external helper
            for i in range(2 ** num_controls):
                new_mottonen.extend(
                    [[prev.active_nodes[num_controls - 1 - control_indices[i]],
                      target_qubit]]
                )
            prev.mottonen_cx_connections.append(new_mottonen)

            # Placeholder for dc_cswap
            new_dc_cswap = [[] for _ in range(prev_layer_size + 2 ** level)]
            prev.dc_cswap_connections.append(new_dc_cswap)
            #prev.dc_cswap_connections.extend(new_dc_cswap)

            return RAAE_StructureResult(
                structure_list=prev.structure_list,
                last_number=prev.structure_list[-1][-1],
                active_nodes=prev.active_nodes + [len(prev.structure_list) - 1],
                node_types=prev.node_types,
                mottonen_cx_connections=prev.mottonen_cx_connections,
                dc_cswap_connections=prev.dc_cswap_connections
            )

        # Even level -> divide-and-conquer style extension
        else:
            prev = self.build(level - 1)
            prev_size = prev.last_number + 1

            new_structure = [[0]]
            for blk in add_number(prev.structure_list, 1):
                new_structure.append([[]] + blk)
            for blk in add_number(prev.structure_list, 1 + prev_size):
                new_structure.append([[]] + blk)

            new_types = [['d']]
            for t in prev.node_types:
                new_types.append([[]] + t)
            for t in prev.node_types:
                new_types.append([[]] + t)

            last_number = 2 * prev.last_number + 2

            new_mottonen = [[]]
            for c in add_number(prev.mottonen_cx_connections, 1):
                new_mottonen.append([[]] + c)
            for c in add_number(prev.mottonen_cx_connections, 1 + len(prev.node_types)):
                new_mottonen.append([[]] + c)

            offset = len(prev.structure_list)
            shifted_outputs = add_number(prev.active_nodes, 1)
            new_dc_cswap = [[[[n, n + offset] for n in shifted_outputs]]]
            for c in add_number(prev.dc_cswap_connections, 1):
                #new_dc_cswap.append([[[]] + c])
                new_dc_cswap.extend([[] + c])
            for c in add_number(prev.dc_cswap_connections, 1 + len(prev.node_types)):
                #new_dc_cswap.append([[[]] + c])
                new_dc_cswap.extend([[] + c])

            return RAAE_StructureResult(
                structure_list=new_structure,
                last_number=last_number,
                active_nodes=[0] + add_number(prev.active_nodes, 1),
                node_types=new_types,
                mottonen_cx_connections=new_mottonen,
                dc_cswap_connections=new_dc_cswap
            )

def flatten_list(item):
    flat = []
    if isinstance(item, list):
        for sub in item:
            flat.extend(flatten_list(sub))
    else:
        flat.append(item)
    return flat

def peel_side(ll):
    left_side = []
    right_side = []

    for sublist in ll:
        # Get the first element as a list (to keep the [0] structure)
        # Using [0:1] ensures it returns [element] if it exists, or [] if empty
        left = sublist[0:1]

        # Get everything from the second element onwards
        right = sublist[1:]

        left_side.append(left)
        right_side.append(right)

    return left_side, right_side

def find_and_peel_d_rightmost(matrix):
    """Find rightmost column containing 'd', peel that column out of each row.

    Returns:
        (column_index, peeled_column, remaining_matrix)
        or (None, None, matrix) if no 'd' is found.
    """
    if not matrix:
        return None, None, matrix

    max_len = max(len(row) for row in matrix)
    found_col_idx = None

    # Scan from rightmost to leftmost column
    for col_idx in range(max_len - 1, -1, -1):
        if any(len(row) > col_idx and row[col_idx] == 'd' for row in matrix):
            found_col_idx = col_idx
            break

    if found_col_idx is None:
        return None, None, matrix

    peeled_column = []
    remaining_matrix = []

    for row in matrix:
        if len(row) > found_col_idx:
            peeled_column.append([row[found_col_idx]])
            remaining_row = row[:found_col_idx] + row[found_col_idx + 1:]
            remaining_matrix.append(remaining_row)
        else:
            peeled_column.append([])
            remaining_matrix.append(row)

    return found_col_idx, peeled_column, remaining_matrix

def find_and_slice_m_leftmost(matrix):
    """Find leftmost column containing 'm' and slice rows around it.

    For rows that have 'm' at that column:
        - Sliced: placeholders up to that column + all 'm' and beyond.
        - Remaining: replaced with single [[]].

    For other rows:
        - Sliced: placeholders up to that column.
        - Remaining: original row.

    Returns:
        (column_index, sliced_matrix, remaining_matrix, sliced_row_indices)
        or (None, None, matrix, []) if no 'm' found.
    """
    if not matrix:
        return None, None, matrix, []

    max_len = max(len(row) for row in matrix)
    found_col_idx = None

    # 1. Find leftmost column containing 'm'
    for col_idx in range(max_len):
        if any(len(row) > col_idx and row[col_idx] == 'm' for row in matrix):
            found_col_idx = col_idx
            break

    if found_col_idx is None:
        return None, None, matrix, []

    sliced_matrix = []
    remaining_matrix = []
    sliced_row_indices = []

    # 2. Process each row
    for row_idx, row in enumerate(matrix):
        has_m_here = len(row) > found_col_idx and row[found_col_idx] == 'm'
        leading_placeholders = [item for item in row[:found_col_idx] if item == []]

        if has_m_here:
            sliced_row_indices.append(row_idx)
            m_part = row[found_col_idx:]
            sliced_matrix.append(leading_placeholders + m_part)
            remaining_matrix.append([[]])
        else:
            sliced_matrix.append(leading_placeholders)
            remaining_matrix.append(row)

    return found_col_idx, sliced_matrix, remaining_matrix, sliced_row_indices


def slice_at_rows(matrix, target_rows):
    """Slice out specific rows, keeping shape with [[]] placeholders.

    For indices in target_rows:
        - sliced gets original row
        - remaining gets [[]]

    For other indices:
        - sliced gets [[]]
        - remaining gets original row
    """
    sliced = []
    remaining = []

    target_rows = set(target_rows)

    for idx, row in enumerate(matrix):
        if idx in target_rows:
            sliced.append(row)
            remaining.append([[]])
        else:
            sliced.append([[]])
            remaining.append(row)

    return sliced, remaining


# =========================
# Generic helpers
# =========================

def is_deeply_empty(obj):
    """Return True if obj is a (possibly nested) list containing no non-list items."""
    if isinstance(obj, list):
        return all(is_deeply_empty(item) for item in obj)
    return False  # Any non-list is considered non-empty


def has_char(target_char, data):
    """Recursively check if target_char exists anywhere in nested list structure."""
    for item in data:
        if isinstance(item, list):
            if has_char(target_char, item):
                return True
        elif isinstance(item, str) and target_char in item:
            return True
    return False

class RAAECircuitBuilder:

    def __init__(self, total_dimension, base_k=2):
        """
        total_dimension : 2^n
        base_k          : base module size (default = 2 qubits)
        """

        self.dim = total_dimension
        self.n_out = int(np.log2(total_dimension))
        if self.n_out % 2 == 0:
          self.n = 2**((self.n_out+2)//2)-2
        else:
          self.n = 2**((self.n_out+3)//2)-3

        self.base_k = base_k

        if 2**self.n_out != total_dimension:
            raise ValueError("Dimension must be power of 2.")

        if base_k < 1 or base_k > self.n:
            raise ValueError("Invalid base module size.")

        self.dc_section_length = int(np.ceil(self.n_out / 2))
        self.mot_section_length = int(np.floor(self.n_out / 2))

        self.RAAE_builder = RAAE_StructureBuilder()
        self.RAAE_structure = self.RAAE_builder.build(self.n_out - 1)

        self.params = ParameterVector('θ', self.dim - 1)

        self.param_ptr = 0
        #print("Uhm")
        self.circuit = self._generate_skeleton()


    # ============================================================
    # SKELETON GENERATION
    # ============================================================
    def _generate_skeleton(self):

        qr = QuantumRegister(self.n, "q")
        qc = QuantumCircuit(qr)

        self.param_ptr = 0

        def raae_build_dc(
          node_types_map,
          mottonen_cx_connections_map,
          dc_cswap_connections_map,
          col_idx = 0):

          # Peel the map
          ntm_l, ntm_r = peel_side(node_types_map)
          mcc_l, mcc_r = peel_side(mottonen_cx_connections_map)
          dcc_l, dcc_r = peel_side(dc_cswap_connections_map)

          # Check availability of the peeled side
          if is_deeply_empty(ntm_l):
              return

          if has_char('d', ntm_l):
              for idx, item in enumerate(ntm_l):
                if item == ['d']:
                  qc.ry(self.params[self.param_ptr], idx)

                  self.last_state = 'd'
                  self.param_ptr += 1

              raae_build_dc(ntm_r, mcc_r, dcc_r, col_idx + 1)
          elif has_char('m', ntm_l):
              raae_build_mot(self.RAAE_structure.node_types,
                             self.RAAE_structure.mottonen_cx_connections)

        def raae_build_mot(
          node_types_map,
          mottonen_cx_connections_map,):
            m_col, ntm_r, ntm_remaining, m_row = find_and_slice_m_leftmost(node_types_map)
            sliced, mottonen_cx_connections_map_remaining = slice_at_rows(mottonen_cx_connections_map, m_row)

            if m_col == None:
                return

            for idx, item in enumerate(ntm_r):
              if is_deeply_empty(item):
                continue
              else:
                  for idx_idx, item_item in enumerate(item):
                    if item_item == 'm':
                      qc.ry(self.params[self.param_ptr], idx)
                      qc.cx(sliced[idx][idx_idx][0], sliced[idx][idx_idx][1])

                      self.param_ptr += 1
                      self.last_state = 'm'

            d_col, ntm_c, self.last_remaining_d_peel = find_and_peel_d_rightmost(self.last_remaining_d_peel)
            if d_col is not None:
                for idx, item in enumerate(ntm_c):
                  if is_deeply_empty(item):
                    continue

                  for idx_idx, item_item in enumerate(item):
                    if item_item == 'd':
                      if self.RAAE_structure.dc_cswap_connections[idx][idx_idx][0] == []:
                          for dc_idx, dc_item in enumerate(self.RAAE_structure.dc_cswap_connections[idx][idx_idx]):
                              if is_deeply_empty(dc_item):
                                continue
                              else:
                                for dc_idx_idx, dc_item_item in enumerate(dc_item):
                                  qc.cswap(
                                      idx,
                                      dc_item_item[0],
                                      dc_item_item[1])
                      else:
                          for dc_idx, dc_item in enumerate(self.RAAE_structure.dc_cswap_connections[idx][idx_idx]):
                              qc.cswap(
                                  idx,
                                  dc_item[0],
                                  dc_item[1])

            raae_build_mot(ntm_remaining, mottonen_cx_connections_map_remaining)


        self.last_state = "d"
        self.last_remaining_d_peel = None
        d_col, ntm_c, self.last_remaining_d_peel = find_and_peel_d_rightmost(self.RAAE_structure.node_types)
        #ipdb.set_trace(context=7)
        raae_build_dc(
            self.RAAE_structure.node_types,
            self.RAAE_structure.mottonen_cx_connections,
            self.RAAE_structure.dc_cswap_connections,
            0
        )

        return qc

    def get_active_nodes(self):
        return self.RAAE_structure.active_nodes

    def get_circuit(self):
        return self.circuit

    def get_circuit_with_measurement(self):
        return measure_selected_qubits(self.get_circuit(), self.get_active_nodes())

def simulate_raae_optimized(input_vector, base_qc, active_nodes, shots=None, noise_model=None):
    """
    Menjalankan simulasi pada sirkuit yang sudah dibangun sebelumnya.
    """
    # 1. Normalisasi
    input_vector = normalize(input_vector)

    # 2. Generate Parameters (Proses Klasik)
    encoder = PAEDataEncoder()
    runner = HierarchicalEncoderRunner(encoder)
    hierarchical_encoding = runner.run(input_vector)
    flat_params = flatten_list(hierarchical_encoding)

    # 3. Assign Parameters ke Sirkuit yang sudah ada
    # Menggunakan inplace=False agar base_qc tetap bersih untuk penggunaan berikutnya
    bound_qc = base_qc.assign_parameters(flat_params)

    # 4. Komputasi/Simulasi
    if shots is not None:
        if noise_model is None:
            backend = AerSimulator()
        else:
            backend = AerSimulator(noise_model=noise_model)
        # Gunakan active_nodes dari structure yang sudah dibuat di luar
        qc_meas = measure_selected_qubits(bound_qc, active_nodes)
        
        job = backend.run(qc_meas, shots=shots)
        counts = job.result().get_counts()

        sim_probabilities = np.zeros(len(input_vector))
        for bitstring, count in counts.items():
            index = int(bitstring[::-1], 2)
            sim_probabilities[index] = count / shots
        sim_amplitudes = np.sqrt(sim_probabilities)
    else:
        # Statevector simulation (Ideal)
        statevector = Statevector.from_instruction(bound_qc)
        sim_amplitudes = np.asarray(statevector.data)
        sim_probabilities = np.abs(sim_amplitudes) ** 2

    return sim_amplitudes, sim_probabilities