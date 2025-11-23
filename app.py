from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import numpy as np
import pandas as pd
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS untuk API

# ============================================
# ROUTING HALAMAN WEB
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tugas1')
def tugas1():
    return render_template('tugas1.html')

@app.route('/jst')
def jst():
    return render_template('jst.html')

@app.route('/fuzzy')
def fuzzy():
    return render_template('fuzzy.html')

@app.route('/genetika')
def genetika():
    return render_template('genetika.html')

@app.route('/tugas2')
def tugas2():
    return render_template('tugas2.html')

@app.route('/tugas3')
def tugas3():
    return render_template('tugas3.html')

# ============================================
# KNAPSACK GENETIC ALGORITHM - DATA & FUNGSI
# ============================================

# Data Masalah Knapsack
items = {
    'A': {'weight': 7, 'value': 5},
    'B': {'weight': 2, 'value': 4},
    'C': {'weight': 1, 'value': 7},
    'D': {'weight': 9, 'value': 2},
}

capacity = 15
item_list = list(items.keys())
n_items = len(item_list)

# Fungsi bantu
def decode(chromosome):
    """Kembalikan list item, total berat, total nilai"""
    total_weight = 0
    total_value = 0
    chosen_items = []
    for gene, name in zip(chromosome, item_list):
        if gene == 1:
            total_weight += items[name]['weight']
            total_value += items[name]['value']
            chosen_items.append(name)
    return chosen_items, total_weight, total_value

def fitness(chromosome):
    """Fungsi fitness dengan penalti berat"""
    _, total_weight, total_value = decode(chromosome)
    if total_weight <= capacity:
        return total_value
    else:
        return 0

def roulette_selection(population, fitnesses):
    """Seleksi roulette wheel"""
    total_fit = sum(fitnesses)
    
    if total_fit == 0:
        return random.choice(population)
    
    pick = random.uniform(0, total_fit)
    current = 0
    for chrom, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return chrom
    return population[-1]

def crossover(p1, p2):
    """Single-point crossover"""
    if len(p1) != len(p2):
        raise ValueError("Parent length mismatch")

    point = random.randint(1, len(p1) - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(chromosome, mutation_rate=0.1):
    """Flip bit dengan probabilitas mutation_rate"""
    return [1 - g if random.random() < mutation_rate else g for g in chromosome]

# Algoritma Genetika Utama
def genetic_algorithm(pop_size=10, generations=10, crossover_rate=0.8, mutation_rate=0.1, elitism=True):
    log = []
    
    # Inisialisasi populasi acak
    population = [[random.randint(0, 1) for _ in range(n_items)] for _ in range(pop_size)]

    for gen in range(generations):
        # Hitung fitness
        fitnesses = [fitness(ch) for ch in population]

        # Catat individu terbaik
        best_index = fitnesses.index(max(fitnesses))
        best_chrom = population[best_index]
        best_fit = fitnesses[best_index]
        best_items, w, v = decode(best_chrom)

        # Simpan ke log
        log.append({
            'generation': gen + 1,
            'chromosome': best_chrom,
            'items': best_items,
            'weight': w,
            'value': v,
            'fitness': best_fit
        })

        # Buat generasi baru
        new_population = []

        # Elitism: pertahankan individu terbaik
        if elitism:
            new_population.append(best_chrom)

        # Reproduksi
        while len(new_population) < pop_size:
            # Seleksi orang tua
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutasi
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            # Tambah ke populasi baru
            new_population.extend([child1, child2])

        # Batasi ukuran populasi
        population = new_population[:pop_size]

    # Ambil hasil akhir
    fitnesses = [fitness(ch) for ch in population]
    best_index = fitnesses.index(max(fitnesses))
    best_chrom = population[best_index]
    best_items, w, v = decode(best_chrom)
    best_fit = fitnesses[best_index]

    return {
        'log': log,
        'final': {
            'chromosome': best_chrom,
            'items': best_items,
            'weight': w,
            'value': v,
            'fitness': best_fit
        }
    }

# ============================================
# API ENDPOINTS UNTUK KNAPSACK GA
# ============================================

@app.route('/api/run_ga', methods=['POST'])
def run_ga():
    """Endpoint untuk menjalankan algoritma genetika"""
    try:
        data = request.get_json()
        
        pop_size = data.get('pop_size', 8)
        generations = data.get('generations', 8)
        crossover_rate = data.get('crossover_rate', 0.8)
        mutation_rate = data.get('mutation_rate', 0.1)
        
        # Set random seed untuk konsistensi (opsional)
        random.seed(42)
        
        # Jalankan algoritma
        result = genetic_algorithm(
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/items', methods=['GET'])
def get_items():
    """Endpoint untuk mendapatkan data items"""
    return jsonify({
        'items': items,
        'capacity': capacity
    })

# ============================================
# TSP GENETIC ALGORITHM - FUNGSI
# ============================================

def route_distance_tsp(route, dist_matrix):
    """Menghitung total jarak dari suatu rute TSP"""
    d = sum(dist_matrix[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))
    return d

def create_individual_tsp(n):
    """Membuat individu (rute) acak sebagai permutasi dari n kota"""
    ind = list(range(n))
    random.shuffle(ind)
    return ind

def initial_population_tsp(size, n):
    """Membuat populasi awal berisi 'size' individu"""
    return [create_individual_tsp(n) for _ in range(size)]

def tournament_selection_tsp(pop, dist_matrix, k):
    """Seleksi turnamen"""
    candidates = random.sample(pop, k)
    return min(candidates, key=lambda ind: route_distance_tsp(ind, dist_matrix))

def ordered_crossover_tsp(p1, p2):
    """Ordered Crossover (OX)"""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    
    p2_idx = 0
    for i in range(n):
        if child[i] == -1:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
            p2_idx += 1
            
    return child

def swap_mutation_tsp(ind):
    """Swap Mutation"""
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]
    return ind

def run_tsp_ga(dist_matrix, cities, pop_size=100, generations=500, tournament_k=5, pc=0.9, pm=0.2, elite_size=1):
    """Menjalankan TSP dengan Genetic Algorithm"""
    n = len(cities)
    
    # Inisialisasi
    pop = initial_population_tsp(pop_size, n)
    best = min(pop, key=lambda ind: route_distance_tsp(ind, dist_matrix))
    best_dist = route_distance_tsp(best, dist_matrix)
    
    history = []
    
    for g in range(generations):
        new_pop = []
        
        # Elitisme
        pop = sorted(pop, key=lambda ind: route_distance_tsp(ind, dist_matrix))
        
        if route_distance_tsp(pop[0], dist_matrix) < best_dist:
            best = pop[0]
            best_dist = route_distance_tsp(best, dist_matrix)
        
        new_pop.extend(pop[:elite_size])
        
        # Reproduksi
        while len(new_pop) < pop_size:
            p1 = tournament_selection_tsp(pop, dist_matrix, tournament_k)
            p2 = tournament_selection_tsp(pop, dist_matrix, tournament_k)
            
            if random.random() < pc:
                child = ordered_crossover_tsp(p1, p2)
            else:
                child = p1[:]
                
            if random.random() < pm:
                child = swap_mutation_tsp(child)
                
            new_pop.append(child)
        
        pop = new_pop
        history.append(best_dist)
    
    # Konversi rute ke nama kota
    best_route = [cities[i] for i in best]
    
    return {
        'best_route': best_route + [best_route[0]],  # Kembali ke kota awal
        'best_distance': float(best_dist),
        'history': history
    }

# ============================================
# API ENDPOINTS UNTUK TSP
# ============================================

@app.route('/api/run_tsp', methods=['POST'])
def run_tsp():
    """Endpoint untuk menjalankan TSP-GA"""
    try:
        # Baca file Excel
        file = request.files['file']
        
        # Read Excel file
        df = pd.read_excel(BytesIO(file.read()), index_col=0)
        cities = list(df.index)
        dist_matrix = df.values.astype(float)
        
        # Get parameters
        pop_size = int(request.form.get('pop_size', 100))
        generations = int(request.form.get('generations', 500))
        tournament_k = int(request.form.get('tournament_k', 5))
        crossover_prob = float(request.form.get('crossover_prob', 0.9))
        mutation_prob = float(request.form.get('mutation_prob', 0.2))
        
        # Set random seed untuk konsistensi
        random.seed(42)
        np.random.seed(42)
        
        # Jalankan TSP-GA
        result = run_tsp_ga(
            dist_matrix=dist_matrix,
            cities=cities,
            pop_size=pop_size,
            generations=generations,
            tournament_k=tournament_k,
            pc=crossover_prob,
            pm=mutation_prob
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# JALANKAN SERVER
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Server Flask telah berjalan!")
    print("=" * 60)
    print("📍 URL: http://localhost:5000")
    print("📝 Halaman Web:")
    print("   - GET  /              : Halaman Utama")
    print("   - GET  /tugas1        : Tugas 1")
    print("   - GET  /tugas2        : Knapsack GA")
    print("   - GET  /tugas3        : TSP GA")
    print("   - GET  /jst           : JST")
    print("   - GET  /fuzzy         : Fuzzy")
    print("   - GET  /genetika      : Genetika")
    print("")
    print("📝 API Endpoints:")
    print("   - POST /api/run_ga    : Jalankan algoritma genetika (Knapsack)")
    print("   - POST /api/run_tsp   : Jalankan algoritma TSP-GA")
    print("   - GET  /api/items     : Dapatkan data items")
    print("=" * 60)
    
    app.run(debug=True, port=5000)