"""Interface local NutriPlan e integração com o RuFaS instalado na pasta pai."""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
RUFAS_ROOT = ROOT.parent
HOST = '127.0.0.1'
PORT = int(os.environ.get('PORT', '8765'))
MAX_BODY = 2_000_000


def number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f'{name}: informe um número válido.')
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name}: informe um número válido.')
    if not math.isfinite(v) or (minimum is not None and v < minimum) or (maximum is not None and v > maximum):
        raise ValueError(f'{name}: valor fora do intervalo permitido.')
    return v


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError('Envie um objeto JSON.')
    animal = payload.get('animal') or {}
    if not isinstance(animal, dict):
        raise ValueError('Dados do animal inválidos.')
    feeds = payload.get('feeds') or []
    if not isinstance(feeds, list) or not (2 <= len(feeds) <= 25):
        raise ValueError('Informe entre 2 e 25 ingredientes.')
    dmi = number(animal.get('dmi'), 'Ingestão de MS', 1, 40)
    cp_min = number(animal.get('cp_min'), 'Proteína bruta mínima', 0, 50)
    ndf_min = number(animal.get('ndf_min'), 'FDN mínima', 0, 90)
    ndf_max = number(animal.get('ndf_max'), 'FDN máxima', 0, 90)
    energy_min = number(animal.get('energy_min'), 'Energia mínima', 0, 120)
    forage_min = number(animal.get('forage_min', 0), 'Volumoso mínimo', 0, 100)
    if ndf_min > ndf_max:
        raise ValueError('A FDN mínima não pode exceder a FDN máxima.')
    clean = []
    for i, fd in enumerate(feeds, 1):
        if not isinstance(fd, dict):
            raise ValueError(f'Ingrediente {i}: formato inválido.')
        name = str(fd.get('name', '')).strip()[:100]
        if not name:
            raise ValueError(f'Ingrediente {i}: informe o nome.')
        dm = number(fd.get('dm'), f'{name} matéria seca (%)', 1, 100)
        cp = number(fd.get('cp'), f'{name} proteína (%)', 0, 100)
        ndf = number(fd.get('ndf'), f'{name} FDN (%)', 0, 100)
        energy = number(fd.get('energy'), f'{name} energia (Mcal/kg MS)', 0, 5)
        price = number(fd.get('price'), f'{name} preço (R$/kg natural)', 0, 10000)
        low = number(fd.get('min', 0), f'{name} mínimo', 0, 40)
        high = number(fd.get('max', 40), f'{name} máximo', 0, 40)
        if low > high:
            raise ValueError(f'{name}: inclusão mínima maior que máxima.')
        number(fd.get('baseline', 0), f'{name} dieta-base', 0, 40)
        clean.append({'name': name, 'dm': dm, 'cp': cp, 'ndf': ndf, 'energy': energy,
                      'price': price, 'min': low, 'max': high,
                      'forage': bool(fd.get('forage')),
                      'source': str(fd.get('source', 'Sem fonte'))[:250],
                      'source_date': str(fd.get('source_date', ''))[:24]})
    return animal, clean, dmi, cp_min, ndf_min, ndf_max, energy_min, forage_min


def optimize(payload):
    animal, feeds, dmi, cp_min, ndf_min, ndf_max, energy_min, forage_min = validate(payload)
    # xi = kg de matéria seca do alimento i por bovino/dia.
    # preço por kg MS = preço por kg natural / (MS% / 100).
    c = [f['price'] / (f['dm'] / 100) for f in feeds]
    A_ub = [
        [-f['cp'] / 100 for f in feeds],
        [-f['ndf'] / 100 for f in feeds],
        [f['ndf'] / 100 for f in feeds],
        [-f['energy'] for f in feeds],
        [-float(f['forage']) for f in feeds],
    ]
    b_ub = [-dmi * cp_min / 100, -dmi * ndf_min / 100,
            dmi * ndf_max / 100, -energy_min, -dmi * forage_min / 100]
    bounds = [(f['min'], f['max']) for f in feeds]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=[[1.0] * len(feeds)],
                     b_eq=[dmi], bounds=bounds, method='highs')
    if not result.success:
        return {'ok': False, 'engine': 'Modelo linear didático (SciPy/HiGHS)',
                'message': ('Não foi possível encontrar dieta viável com esses dados. '
                            'Revise limites de ingredientes, proteína, FDN, energia e consumo de MS.'),
                'solver_status': int(result.status)}
    x = [max(0.0, float(v)) if float(v) >= -1e-7 else float(v) for v in result.x]
    if any(not math.isfinite(v) or v < 0 for v in x):
        raise ValueError('O solucionador retornou quantidades inválidas.')
    ingredients = []
    for f, qty, cost_per_dm in zip(feeds, x, c):
        ingredients.append({'name': f['name'], 'kg_dm': qty, 'kg_natural': qty / (f['dm'] / 100),
                            'percent_dm': (qty / dmi) * 100, 'cost': qty * cost_per_dm,
                            'source': f['source'], 'source_date': f['source_date']})
    cp_kg = sum(qty * f['cp'] / 100 for qty, f in zip(x, feeds))
    ndf_kg = sum(qty * f['ndf'] / 100 for qty, f in zip(x, feeds))
    en = sum(qty * f['energy'] for qty, f in zip(x, feeds))
    forage_kg = sum(qty for qty, f in zip(x, feeds) if f['forage'])
    baseline = payload.get('baseline')
    baseline_cost = None
    baseline_status = 'Dieta-base ausente ou incompleta'
    if isinstance(baseline, list) and len(baseline) == len(feeds):
        try:
            bs = [number(v, 'Dieta base', 0, 40) for v in baseline]
            violations = []
            if abs(sum(bs) - dmi) > 0.05:
                violations.append('matéria seca')
            if sum(q * f['cp'] / 100 for q, f in zip(bs, feeds)) < dmi * cp_min / 100 - 1e-4:
                violations.append('proteína bruta')
            baseline_ndf = sum(q * f['ndf'] / 100 for q, f in zip(bs, feeds))
            if not (dmi * ndf_min / 100 - 1e-4 <= baseline_ndf <= dmi * ndf_max / 100 + 1e-4):
                violations.append('FDN')
            if sum(q * f['energy'] for q, f in zip(bs, feeds)) < energy_min - 1e-4:
                violations.append('energia')
            if sum(q for q, f in zip(bs, feeds) if f['forage']) < dmi * forage_min / 100 - 1e-4:
                violations.append('volumosos')
            if any(q < f['min'] - 1e-4 or q > f['max'] + 1e-4 for q, f in zip(bs, feeds)):
                violations.append('limites de ingredientes')
            if violations:
                baseline_status = 'Base não atende: ' + ', '.join(violations)
            else:
                baseline_cost = sum(a * b for a, b in zip(bs, c))
                baseline_status = 'Base viável no modelo didático'
        except ValueError:
            baseline_status = 'Valores inválidos na dieta-base'
    return {'ok': True, 'engine': 'Modelo linear didático (SciPy/HiGHS)',
            'cost_day': float(result.fun), 'cost_month': float(result.fun * 30),
            'dmi': dmi, 'animal': {'weight': animal.get('weight'), 'milk': animal.get('milk')},
            'baseline_cost': baseline_cost, 'baseline_status': baseline_status,
            'saving_percent': (100 * (baseline_cost - result.fun) / baseline_cost) if baseline_cost and baseline_cost > 0 else None,
            'metrics': {'cp_pct': cp_kg / dmi * 100, 'ndf_pct': ndf_kg / dmi * 100,
                        'energy': en, 'forage_pct': forage_kg / dmi * 100},
            'targets': {'cp_pct': cp_min, 'ndf_min': ndf_min, 'ndf_max': ndf_max,
                        'energy': energy_min, 'forage_pct': forage_min},
            'ingredients': ingredients,
            'limitations': ('Modelo didático: as exigências foram informadas pelo usuário. '
                            'PB não equivale à proteína metabolizável; não contempla minerais, '
                            'aminoácidos, digestibilidade dinâmica, ingestão prevista ou produção real. '
                            'Não é o otimizador nativo do RuFaS.')}


def rufas_native(payload):
    """Executa o *RationOptimizer real* com Feed/NutritionRequirements COMPLETOS.

    Formato JSON exigido para evitar inventar propriedades nutricionais:
      {"nutrient_standard": "NRC", "animal_combination": "lac_cow",
       "body_weight": 600, "dmi_initial": 20, "protein_initial": 1800,
       "requirements": {todos os campos reais de NutritionRequirements},
       "feeds": [{todos os campos reais de Feed, com rufas_id}, ...]}

    Estes registros são obtidos do próprio RuFaS / biblioteca nutricional validada.
    O preço `purchase_cost` deve estar na mesma unidade de decisão (por kg de MS).
    """
    if not isinstance(payload, dict):
        raise ValueError('Envie um objeto JSON.')
    if str(RUFAS_ROOT) not in sys.path:
        sys.path.insert(0, str(RUFAS_ROOT))
    try:
        from RUFAS.biophysical.animal.ration.ration_optimizer import RationOptimizer
        from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
        from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
        from RUFAS.data_structures.feed_storage_to_animal_connection import (
            Feed, NutrientStandard, FeedCategorization, FeedComponentType)
        from RUFAS.units import MeasurementUnits
    except ImportError as exc:
        raise RuntimeError('RuFaS indisponível no ambiente Python atual.') from exc
    if str(payload.get('nutrient_standard', 'NRC')).upper() != 'NRC':
        raise ValueError('Esta interface aceita somente NRC com registros Feed completos. NASEM exige NASEMFeed.')
    raw_feeds = payload.get('feeds')
    if not isinstance(raw_feeds, list) or not (2 <= len(raw_feeds) <= 25):
        raise ValueError('Informe entre 2 e 25 registros Feed completos.')
    field_set = {f.name for f in fields(Feed)}
    required_feed_fields = set(field_set)
    feed_list = []
    ids = set()
    for index, item in enumerate(raw_feeds, 1):
        if not isinstance(item, dict):
            raise ValueError(f'Feed {index}: formato inválido.')
        missing = required_feed_fields - set(item)
        if missing:
            raise ValueError(f'Feed {index}: campos ausentes: {", ".join(sorted(missing))}.')
        data = {key: item[key] for key in field_set}
        feed_id = data['rufas_id']
        if isinstance(feed_id, bool) or not isinstance(feed_id, int) or feed_id in ids:
            raise ValueError(f'Feed {index}: rufas_id deve ser inteiro único.')
        ids.add(feed_id)
        for key in field_set - {'rufas_id', 'Fd_Category', 'feed_type', 'units', 'is_fat', 'is_wetforage'}:
            data[key] = number(data[key], f'Feed {feed_id} {key}', 0)
        for key in ('is_fat', 'is_wetforage'):
            if not isinstance(data[key], bool):
                raise ValueError(f'Feed {feed_id}: {key} deve ser booleano.')
        if not 0 < data['DM'] <= 100 or data['lower_limit'] > data['limit']:
            raise ValueError(f'Feed {feed_id}: matéria seca ou limites inválidos.')
        for key, EnumType in [('Fd_Category', FeedCategorization), ('feed_type', FeedComponentType), ('units', MeasurementUnits)]:
            if not isinstance(data[key], EnumType):
                data[key] = next((e for e in EnumType if e.value == data[key] or e.name == data[key]), None)
                if data[key] is None:
                    raise ValueError(f'Feed {item.get("rufas_id")}: enum {key} inválido.')
        feed_list.append(Feed(**data))
    nr = payload.get('requirements') or {}
    if not isinstance(nr, dict):
        raise ValueError('NutritionRequirements deve ser um objeto.')
    required_nr = {f.name for f in fields(NutritionRequirements)}
    if not required_nr.issubset(nr):
        raise ValueError('Dados de NutritionRequirements incompletos. Campos ausentes: ' + ', '.join(sorted(required_nr - set(nr))))
    # EssentialAminoAcidRequirements precisa ser reconstruído como dataclass.
    from RUFAS.biophysical.animal.ration.amino_acid import EssentialAminoAcidRequirements
    nr = dict(nr)
    amino = nr.get('essential_amino_acids')
    amino_names = {f.name for f in fields(EssentialAminoAcidRequirements)}
    if not isinstance(amino, dict) or not amino_names.issubset(amino):
        raise ValueError('essential_amino_acids incompleto.')
    nr['essential_amino_acids'] = EssentialAminoAcidRequirements(**{
        key: number(amino[key], key, 0) for key in amino_names})
    for key in required_nr - {'essential_amino_acids'}:
        nr[key] = number(nr[key], key, 0)
    requirements = NutritionRequirements(**{key: nr[key] for key in required_nr})
    std = NutrientStandard.NRC
    try:
        group = AnimalCombination(str(payload.get('animal_combination', 'lac_cow')))
    except ValueError as exc:
        raise ValueError('Categoria RuFaS inválida.') from exc
    optimizer = RationOptimizer()
    solution, config = optimizer.attempt_optimization(
        nutrient_standard=std,
        pen_average_body_weight=number(payload.get('body_weight'), 'Peso', 20, 2000),
        pen_average_enteric_methane=None, pen_average_urine_nitrogen=None,
        requirements=requirements,
        initial_dry_matter_requirement=number(payload.get('dmi_initial'), 'DMI', 1, 40),
        initial_protein_requirement=number(payload.get('protein_initial'), 'Proteína metabolizável', 0, 10000),
        pen_available_feeds=feed_list, animal_combination=group,
    )
    if not solution.success:
        return {'ok': False, 'engine': 'RuFaS RationOptimizer (nativo)',
                'message': 'Não foi encontrada solução viável para as entradas e restrições fornecidas.',
                'solver_status': int(solution.status)}
    sorted_feeds = sorted(feed_list, key=lambda f: f.rufas_id)
    quantities = [float(v) for v in solution.x]
    if any(not math.isfinite(v) or v < -1e-6 for v in quantities):
        raise RuntimeError('O RuFaS retornou quantidades inválidas.')
    quantities = [max(0.0, v) for v in quantities]
    constraints = []
    for constraint in optimizer._select_constraints(group):
        value = float(constraint['fun'](solution.x, config))
        constraints.append({'name': constraint['fun'].__name__, 'slack': value if math.isfinite(value) else None,
                            'satisfied': math.isfinite(value) and value >= -1e-4})
    if not all(c['satisfied'] for c in constraints):
        return {'ok': False, 'engine': 'RuFaS RationOptimizer (nativo)',
                'message': 'O solucionador terminou, mas uma ou mais restrições não foram atendidas.',
                'constraints': constraints}
    cost = sum(q * f.purchase_cost for f, q in zip(sorted_feeds, quantities))
    feed_names = {f['rufas_id']: str(f.get('name') or f['rufas_id'])[:100] for f in raw_feeds}
    return {'ok': True, 'engine': 'RuFaS RationOptimizer (nativo)',
            'cost_day': cost, 'cost_month': cost * 30,
            'dmi': sum(quantities), 'constraints': constraints,
            'ingredients': [{'id': f.rufas_id, 'name': feed_names[f.rufas_id],
                             'kg_dm': qty, 'kg_natural': qty / (f.DM / 100),
                             'percent_dm': 100 * qty / sum(quantities) if sum(quantities) else 0,
                             'cost': qty * f.purchase_cost}
                            for f, qty in zip(sorted_feeds, quantities)],
            'message': 'Calculado pelo RationOptimizer do RuFaS. Composição, exigências e preços são responsabilidade da fonte informada.'}


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            data = (ROOT / 'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/styles.css':
            data = (ROOT / 'styles.css').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/css; charset=utf-8')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/api/health':
            self._send({'ok': True, 'didactic': 'SciPy/HiGHS',
                        'native': (RUFAS_ROOT / 'RUFAS' / 'biophysical' / 'animal' / 'ration' / 'ration_optimizer.py').is_file()})
        else:
            self._send({'error': 'Rota não encontrada.'}, 404)

    def do_POST(self):
        if self.path not in ('/api/optimize', '/api/rufas-native'):
            return self._send({'error': 'Rota não encontrada.'}, 404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > MAX_BODY:
                return self._send({'error': 'Tamanho de requisição inválido.'}, 413)
            payload = json.loads(self.rfile.read(size))
            data = optimize(payload) if self.path == '/api/optimize' else rufas_native(payload)
            self._send(data)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send({'error': str(exc)}, 422)
        except Exception:
            traceback.print_exc()
            self._send({'error': 'Falha interna ao executar a otimização. Consulte o terminal local.'}, 500)


if __name__ == '__main__':
    print(f'RuFaS Dietas - abrir http://{HOST}:{PORT}')
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
