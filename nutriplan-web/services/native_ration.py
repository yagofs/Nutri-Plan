"""Native NASEM feed loading and ration optimization through installed RuFaS."""
from __future__ import annotations

import csv
import math
from dataclasses import fields
from pathlib import Path

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.data_types.animal_types import AnimalType
from RUFAS.biophysical.animal.ration.ration_optimizer import RationOptimizer
from RUFAS.data_structures.feed_storage_to_animal_connection import (
    FeedCategorization, FeedComponentType, NASEMFeed, NutrientStandard,
)
from RUFAS.units import MeasurementUnits

from services.nutrition import calculate
from services.numbers import parse_number
from services.emissions import (
    calculate_candidate_ration_emissions, previous_ration_ndf_percent,
)
from RUFAS.biophysical.animal.nutrients.nutrition_supply_calculator import FeedInRation, NutritionSupplyCalculator
from RUFAS.biophysical.animal.animal_module_constants import AnimalModuleConstants
from services.prices import comparison_status, default_price, resolve_price
from services.milk_potential import (
    FLOOR_TOLERANCE_KG_DAY, evaluate as evaluate_milk_potential, resolve_reference,
)

FEED_CSV = Path(__file__).resolve().parents[2] / 'input/data/feed/NASEM_Comp_with_TDN.csv'
ENUMS = {'Fd_Category': FeedCategorization, 'feed_type': FeedComponentType, 'units': MeasurementUnits}
BOOLEANS = {'is_fat', 'is_wetforage'}
OPERATIONS = {'amount_available', 'on_farm_cost', 'purchase_cost', 'buffer'}

# Structural mapping is explicit so each stage uses its native pen group.
ANIMAL_MODELS = {
    'multiparous_lactating_cow': (AnimalType.LAC_COW, AnimalCombination.LAC_COW, 'lactating'),
    'first_lactation_cow': (AnimalType.LAC_COW, AnimalCombination.LAC_COW, 'lactating'),
    'heifer_I': (AnimalType.HEIFER_I, AnimalCombination.GROWING, 'growth'),
    'heifer_II': (AnimalType.HEIFER_II, AnimalCombination.GROWING, 'growth'),
    'heifer_III': (AnimalType.HEIFER_III, AnimalCombination.CLOSE_UP, 'growth'),
    'dry_cow': (AnimalType.DRY_COW, AnimalCombination.CLOSE_UP, 'dry'),
}
OPTIMIZATION_READY = {
    'multiparous_lactating_cow', 'first_lactation_cow',
    'heifer_I', 'heifer_II', 'heifer_III', 'dry_cow',
}
LACTATING_CATEGORIES = {'multiparous_lactating_cow', 'first_lactation_cow'}


class DeterministicRationOptimizer(RationOptimizer):
    """Start native optimization from the same feasible bounds for identical inputs."""

    def _build_initial_value(self, previous_ration, ration_config):
        if previous_ration is not None:
            return super()._build_initial_value(previous_ration, ration_config)
        target = ration_config.initial_dry_matter_requirement * ration_config.dmi_requirement_boost
        return _allocate_candidate_seed(ration_config.feeds_used, target)


class CandidateRationOptimizer(DeterministicRationOptimizer):
    """Use native lactating-cow equations with candidate-specific NDF and emissions."""

    def __init__(self, animal_payload, calculate_candidate_emissions=True, seed=None, *,
                 candidate_ndf=True, calculation_type='lactating'):
        super().__init__()
        self.animal_payload = animal_payload
        self.calculate_candidate_emissions = calculate_candidate_emissions
        self.candidate_seed = seed
        self.candidate_ndf = candidate_ndf
        self.calculation_type = calculation_type

    def _build_initial_value(self, previous_ration, ration_config):
        if previous_ration is None and self.candidate_seed is not None:
            return list(self.candidate_seed)
        return super()._build_initial_value(previous_ration, ration_config)

    def set_constraints(self, ration_config):
        super().set_constraints(ration_config)
        for constraints in (self.cow_constraints, self.heifer_constraints):
            for constraint in constraints:
                function_name = constraint['fun'].__name__
                if function_name == 'NASEM_net_energy_constraint':
                    constraint['fun'] = self._candidate_net_energy_constraint
                elif self.candidate_ndf and function_name == 'protein_constraint_lower':
                    constraint['fun'] = self._candidate_protein_lower
                elif self.candidate_ndf and function_name == 'protein_constraint_upper':
                    constraint['fun'] = self._candidate_protein_upper

    @staticmethod
    def _candidate_feeds(decision_vector, ration_configuration):
        return RationOptimizer.convert_decision_vec_to_feeds(ration_configuration, decision_vector)

    @staticmethod
    def _candidate_ndf(feeds):
        dry_matter = sum(feed.amount for feed in feeds)
        if dry_matter <= 0 or not math.isfinite(dry_matter):
            return math.nan
        ndf_supply = NutritionSupplyCalculator._calculate_nutritive_content(feeds, 'NDF')
        return 100 * ndf_supply / dry_matter

    def _candidate_requirements(self, ndf_percentage):
        parameters = dict(self.animal_payload.get('parameters') or {})
        parameters['ndf_percentage'] = ndf_percentage
        result = calculate({**self.animal_payload, 'parameters': parameters}, include_native=True)
        if result.get('status') != 'success':
            return None
        return result['_native_requirements']

    def _candidate_emission_values(self, feeds, ration_configuration):
        if self.calculate_candidate_emissions:
            emissions = calculate_candidate_ration_emissions(feeds, {
                'animal_category': self.animal_payload['animal_category'],
                'body_weight_kg': ration_configuration.pen_average_body_weight,
                'parameters': self.animal_payload['parameters'],
            }, self.calculation_type)
            return emissions['enteric_methane_g_day'], emissions['urine_nitrogen_kg_day']
        return (ration_configuration.pen_average_enteric_methane,
                ration_configuration.pen_average_urine_nitrogen)

    def _candidate_net_energy_constraint(self, decision_vector, ration_configuration):
        feeds = self._candidate_feeds(decision_vector, ration_configuration)
        dry_matter = sum(decision_vector)
        total_starch = sum(feed.amount * feed.info.starch / 100 for feed in feeds)
        methane, urine_nitrogen = self._candidate_emission_values(feeds, ration_configuration)
        metabolizable_energy = NutritionSupplyCalculator.calculate_NASEM_metabolizable_energy(
            feeds=feeds, dry_matter_intake=dry_matter,
            body_weight=ration_configuration.pen_average_body_weight,
            total_starch=total_starch, enteric_methane=methane,
            urinary_nitrogen=urine_nitrogen,
        )
        net_energy = NutritionSupplyCalculator.calculate_NASEM_net_energy(metabolizable_energy)
        return net_energy - ration_configuration.animal_requirements.total_energy_requirement

    def _candidate_protein_parameters(self, decision_vector, ration_configuration):
        feeds = self._candidate_feeds(decision_vector, ration_configuration)
        ndf_percentage = self._candidate_ndf(feeds)
        if not math.isfinite(ndf_percentage):
            return math.nan, math.nan
        requirements = self._candidate_requirements(ndf_percentage)
        if requirements is None:
            return math.nan, math.nan
        protein_supply, _ = RationOptimizer._calculate_protein_constraint_parameters(
            decision_vector, ration_configuration, use_initial_requirement=False
        )
        return protein_supply, requirements.metabolizable_protein

    def _candidate_protein_lower(self, decision_vector, ration_configuration):
        supply, requirement = self._candidate_protein_parameters(decision_vector, ration_configuration)
        return supply - requirement

    def _candidate_protein_upper(self, decision_vector, ration_configuration):
        supply, requirement = self._candidate_protein_parameters(decision_vector, ration_configuration)
        return requirement * AnimalModuleConstants.PROTEIN_UPPER_LIMIT_FACTOR - supply


def animal_model(animal_category):
    """Resolve physiology, RuFaS pen group, and requirement pathway explicitly."""
    try:
        return ANIMAL_MODELS[animal_category]
    except (KeyError, TypeError) as error:
        raise ValueError(f'Unsupported animal category: {animal_category}.') from error


def feed_library():
    """Read the installed RuFaS NASEM feed library, rejecting duplicate identifiers."""
    with FEED_CSV.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    library = {}
    for row in rows:
        identifier = int(row['rufas_id'])
        if identifier in library:
            raise ValueError(f'Duplicate RuFaS feed identifier: {identifier}')
        library[identifier] = row
    return library


# Curated feeds used by the automatic selection mode. Frozen on purpose so the
# optimizer's candidate pool stays predictable and scientifically reviewed.
RECOMMENDED_FEED_IDS = frozenset({
    52, 51, 110, 104, 106, 103,            # volumosos
    154, 58, 4, 33, 49, 44, 193,           # energeticos
    60, 181, 31, 164, 167, 42, 25,         # proteicos
    230, 234, 258, 231, 229,               # minerais / corretivos
})

# Feeds that are not market inputs for a dairy diet and never belong to a group.
EXCLUDED_FEED_CATEGORIES = {'Calf Liquid Feed'}
EXCLUDED_FEED_NAME_PREFIXES = ('calf starter', 'farm es ')

FEED_GROUPS = ('forage', 'energy', 'protein', 'mineral')


def _num(row, key):
    try:
        value = float(row[key])
        return value
    except (KeyError, TypeError, ValueError):
        return None


def _is_excluded_feed(row):
    category = str(row.get('Fd_Category') or '')
    name = str(row.get('Name') or '').strip().lower()
    if category in EXCLUDED_FEED_CATEGORIES:
        return True
    return any(name.startswith(prefix) for prefix in EXCLUDED_FEED_NAME_PREFIXES)


def feed_group(row):
    """Classify a feed into one of the four selection groups by composition.

    Priority: mineral -> protein -> energy -> forage, so every market feed lands
    in exactly one group.
    """
    category = str(row.get('Fd_Category') or '')
    feed_type = str(row.get('feed_type') or '')
    ndf = _num(row, 'NDF') or 0.0
    cp = _num(row, 'CP') or 0.0
    starch = _num(row, 'starch') or 0.0
    tdn = _num(row, 'TDN') or 0.0
    ee = _num(row, 'EE') or 0.0
    if category == 'Vitamin/Mineral' or feed_type == 'Mineral':
        return 'mineral'
    if feed_type != 'Forage' and cp >= 25.0:
        return 'protein'
    if feed_type != 'Forage' and (starch >= 20.0 or tdn >= 75.0 or ee >= 12.0):
        return 'energy'
    return 'forage'


def automatic_candidates():
    """Build the optimizer's candidate list from the curated pool.

    Only recommended feeds that carry a preloaded price take part; inclusion
    bounds come from the NASEM feed library.
    """
    library = feed_library()
    candidates = []
    for identifier in sorted(RECOMMENDED_FEED_IDS):
        row = library.get(identifier)
        if row is None or default_price(identifier) is None:
            continue
        candidates.append({
            'rufas_id': identifier,
            'available': True,
            'minimum_kg_dm': float(row['lower_limit']) if row.get('lower_limit') else 0.0,
            'maximum_kg_dm': float(row['limit']) if row.get('limit') else 100.0,
        })
    return candidates


def available_feeds():
    feeds = []
    for identifier, row in feed_library().items():
        feeds.append({
            'id': identifier,
            'name': row['Name'],
            'category': row['Fd_Category'],
            'group': None if _is_excluded_feed(row) else feed_group(row),
            'recommended': identifier in RECOMMENDED_FEED_IDS,
            'dry_matter_percent': float(row['DM']),
            'crude_protein_percent_dm': float(row['CP']) if row['CP'] else None,
            'ndf_percent_dm': float(row['NDF']) if row['NDF'] else None,
            'starch_percent_dm': float(row['starch']) if row['starch'] else None,
            'fat_percent_dm': float(row['EE']) if row['EE'] else None,
            'lower_limit': float(row['lower_limit']) if row.get('lower_limit') else 0.0,
            'limit': float(row['limit']) if row.get('limit') else 100.0,
            'default_price': default_price(identifier),
            'composition_source': str(FEED_CSV.relative_to(FEED_CSV.parents[3])),
        })
    return feeds


def build_feed(item, library=None):
    """Use original NASEM composition and explicit UI price/bounds in kg dry matter."""
    library = library or feed_library()
    try:
        identifier = int(item['rufas_id'])
        row = library[identifier]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Select a valid RuFaS NASEM feed identifier.') from error
    try:
        price_record = resolve_price(item, library)
        if price_record is None:
            raise ValueError('Price is required.')
        price_as_fed = price_record['price_as_fed']
        minimum = parse_number(item.get('minimum_kg_dm', 0), 'minimum_kg_dm', minimum=0)
        maximum = parse_number(item['maximum_kg_dm'], 'maximum_kg_dm', minimum=0)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Feed {identifier}: invalid price or inclusion bounds.') from error
    dry_matter_fraction = float(row['DM']) / 100
    if not all(math.isfinite(value) for value in (price_as_fed, minimum, maximum)) or price_as_fed < 0 or dry_matter_fraction <= 0 or minimum < 0 or maximum < minimum:
        raise ValueError(f'Feed {identifier}: invalid dry matter, price, or bounds.')
    data = {'rufas_id': identifier, 'amount_available': 0.0,
            'purchase_cost': price_as_fed / dry_matter_fraction,
            'on_farm_cost': 0.0, 'buffer': 0.0,
            'lower_limit': minimum, 'limit': maximum}
    for field in fields(NASEMFeed):
        key = field.name
        if key in data:
            continue
        raw = row.get(key)
        if raw is None or raw == '':
            raise ValueError(f'Feed {identifier}: missing NASEM field {key}.')
        if key in ENUMS:
            data[key] = ENUMS[key](raw)
        elif key in BOOLEANS:
            if raw.upper() not in {'TRUE', 'FALSE'}:
                raise ValueError(f'Feed {identifier}: invalid {key}.')
            data[key] = raw.upper() == 'TRUE'
        elif key == 'Name':
            data[key] = raw
        else:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f'Feed {identifier}: invalid {key}.')
            data[key] = value
    return NASEMFeed(**data)


def prepare_animal_payload(payload):
    """Derive previous-ration NDF and reject conflicting manual NDF."""
    if payload.get('previous_ration') is not None:
        library = feed_library()
        derived_ndf = previous_ration_ndf_percent(payload['previous_ration'], library, build_feed)
        parameters = dict(payload.get('parameters') or {})
        supplied_ndf = parameters.get('ndf_percentage')
        if supplied_ndf is not None and supplied_ndf != '':
            try:
                difference = abs(parse_number(supplied_ndf, 'ndf_percentage', minimum=0, maximum=100) - derived_ndf)
            except (TypeError, ValueError) as error:
                raise ValueError('Invalid previous-ration NDF value.') from error
            if not math.isfinite(difference) or difference > 1e-6:
                raise ValueError('Previous-ration NDF does not match the RuFaS feed composition.')
        parameters['ndf_percentage'] = derived_ndf
        payload = {**payload, 'parameters': parameters}
    return payload


def _allocate_candidate_seed(feeds, target_dry_matter):
    """Distribute the native DMI target across available bounds for a deterministic solver start."""
    lower = [float(feed.lower_limit) for feed in feeds]
    upper = [float(feed.limit) for feed in feeds]
    lower_total = sum(lower)
    capacity = [max(0.0, high - low) for low, high in zip(lower, upper)]
    total_capacity = sum(capacity)
    if not math.isfinite(target_dry_matter) or target_dry_matter <= 0:
        raise ValueError('RuFaS returned an invalid dry matter intake requirement.')
    target = min(max(target_dry_matter, lower_total), sum(upper))
    if total_capacity <= 0:
        return lower
    fraction = (target - lower_total) / total_capacity
    return [low + fraction * available for low, available in zip(lower, capacity)]


def _payload_with_ndf(payload, ndf_percentage):
    parameters = dict(payload.get('parameters') or {})
    parameters['ndf_percentage'] = ndf_percentage
    return {**payload, 'parameters': parameters}


def _explicit_emissions(payload):
    try:
        methane = parse_number(payload['enteric_methane_g_day'], 'enteric_methane_g_day', minimum=0)
        urine_nitrogen = parse_number(payload['urine_nitrogen_kg_day'], 'urine_nitrogen_kg_day', minimum=0)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid methane or urinary nitrogen measurement.') from error
    result = {'enteric_methane_g_day': methane, 'urine_nitrogen_kg_day': urine_nitrogen,
              'provenance': 'user_supplied'}
    source = payload.get('emission_source')
    if isinstance(source, str) and source.strip():
        result['source_reference'] = source.strip()
    return result


CONSTRAINT_UNITS = {
    'calcium_constraint': 'g/day',
    'phosphorus_constraint': 'g/day',
    'protein_constraint_lower': 'g/day',
    'protein_constraint_upper': 'g/day',
    'NDF_constraint_lower': '% DM',
    'NDF_constraint_upper': '% DM',
    'forage_NDF_constraint': '% DM',
    'fat_constraint': '% DM',
    'DMI_constraint_lower': 'kg/day',
    'DMI_constraint_upper': 'kg/day',
    'NASEM_net_energy_constraint': 'Mcal/day',
}


def _enrich_constraints(constraints, requirements, configuration):
    """Attach the solver bound and the achieved supply for each constraint."""
    fraction = getattr(
        configuration, 'dmi_constraint_fraction',
        AnimalModuleConstants.DMI_CONSTRAINT_FRACTION,
    )
    for item in constraints:
        name = item['name']
        slack = item['slack']
        minimum = maximum = None
        if name == 'calcium_constraint':
            minimum = requirements.calcium
        elif name == 'phosphorus_constraint':
            minimum = max(requirements.phosphorus, requirements.process_based_phosphorus)
        elif name == 'protein_constraint_lower':
            minimum = requirements.metabolizable_protein
        elif name == 'protein_constraint_upper':
            maximum = requirements.metabolizable_protein * AnimalModuleConstants.PROTEIN_UPPER_LIMIT_FACTOR
        elif name == 'NDF_constraint_lower':
            minimum = AnimalModuleConstants.MINIMUM_RATION_NDF
        elif name == 'NDF_constraint_upper':
            maximum = AnimalModuleConstants.MAXIMUM_RATION_NDF
        elif name == 'forage_NDF_constraint':
            minimum = AnimalModuleConstants.MINIMUM_RATION_FORAGE_NDF
        elif name == 'fat_constraint':
            maximum = AnimalModuleConstants.MAXIMUM_RATION_FAT
        elif name == 'DMI_constraint_lower':
            minimum = requirements.dry_matter * (1 - fraction)
        elif name == 'DMI_constraint_upper':
            maximum = requirements.dry_matter * (1 + fraction)
        elif name == 'NASEM_net_energy_constraint':
            minimum = requirements.total_energy_requirement
        if not math.isfinite(slack) or (minimum is None and maximum is None):
            supply = None
        elif minimum is not None:
            supply = minimum + slack
        else:
            supply = maximum - slack
        item['minimum'] = minimum
        item['maximum'] = maximum
        item['supply'] = supply
        item['unit'] = CONSTRAINT_UNITS.get(name)


def optimize_native(payload):
    """Attempt native NASEM ration optimization with candidate-specific dependencies when available."""
    if payload.get('previous_ration') == []:
        payload = {**payload, 'previous_ration': None}
    category = payload.get('animal_category')
    try:
        _, group, calculation_type = animal_model(category)
    except ValueError as error:
        return {'ok': False, 'status': 'unsupported_category', 'message': str(error)}
    if category not in OPTIMIZATION_READY:
        return {'ok': False, 'status': 'unsupported_category',
                'message': 'This animal category is not enabled for native ration formulation.'}
    is_lactating = calculation_type == 'lactating'

    has_previous_ration = payload.get('previous_ration') is not None
    emission_keys = ('enteric_methane_g_day', 'urine_nitrogen_kg_day')
    missing_emissions = [key for key in emission_keys
                         if payload.get(key) is None or str(payload[key]).strip() == '']
    if len(missing_emissions) == 1:
        return {'ok': False, 'status': 'incomplete_input', 'error_code': 'emissions_pair_required',
                'missing_inputs': missing_emissions,
                'message': 'Provide both methane and urinary nitrogen, or leave both blank for automatic estimation.'}
    explicit_emissions = _explicit_emissions(payload) if not missing_emissions else None

    parameters = dict(payload.get('parameters') or {})
    allow_below_reference = payload.get('allow_below_reference') is True
    explore_alternatives = payload.get('explore_alternatives') is True
    reference_milk = None
    reference_source = None
    if is_lactating:
        reference_milk, reference_source = resolve_reference(payload)
        target_raw = parameters.get('milk_production')
        if target_raw is not None and not (isinstance(target_raw, str) and not target_raw.strip()):
            target_milk = parse_number(target_raw, 'milk_production', minimum=0)
            if not allow_below_reference and target_milk + FLOOR_TOLERANCE_KG_DAY < reference_milk:
                return {'ok': False, 'status': 'invalid_input',
                        'error_code': 'milk_target_below_reference',
                        'message': 'The milk target cannot be below the current production reference.'}
    supplied_ndf = parameters.get('ndf_percentage')
    candidate_ndf_mode = not has_previous_ration and supplied_ndf in (None, '')

    if candidate_ndf_mode and not is_lactating:
        ndf = 30.0
        result = None
        for _ in range(6):
            trial = {**payload, 'parameters': {**parameters, 'ndf_percentage': ndf}}
            result = optimize_native(trial)
            if not result.get('ok'):
                return result
            ndf_supply = (result.get('nutrient_supply') or {}).get('ndf', {}).get('value')
            dry_matter = result.get('dmi')
            if not (isinstance(ndf_supply, (int, float)) and math.isfinite(ndf_supply)
                    and isinstance(dry_matter, (int, float)) and math.isfinite(dry_matter)
                    and dry_matter > 0):
                return {'ok': False, 'status': 'calculation_error',
                        'message': 'Optimized ration NDF is not finite.'}
            final_ndf = 100 * ndf_supply / dry_matter
            if abs(final_ndf - ndf) <= 1e-3:
                result['ration_context'] = {
                    'ndf_percentage': final_ndf,
                    'dry_matter_intake_kg_day': dry_matter,
                    'source': 'RuFaS optimized feed composition',
                }
                return result
            ndf = final_ndf
        return {'ok': False, 'status': 'calculation_error',
                'message': 'Candidate NDF did not converge for the non-lactating ration.'}

    if has_previous_ration:
        payload = prepare_animal_payload(payload)

    selected = payload.get('feeds')
    if selected is None:
        selected = []
    if not isinstance(selected, list):
        raise ValueError('Feeds must be a list.')
    selection_mode = 'manual'
    if not selected:
        # No explicit selection: let the optimizer choose among the curated pool,
        # using each feed's preloaded price and NASEM inclusion bounds.
        selection_mode = 'automatic'
        selected = automatic_candidates()
        if len(selected) < 2:
            return {'ok': False, 'status': 'incomplete_input',
                    'message': 'No priced feeds are available for automatic selection.'}
    elif len(selected) < 2:
        return {'ok': False, 'status': 'incomplete_input', 'message': 'Select at least two NASEM feeds.'}
    library = feed_library()
    if any(not isinstance(item, dict) for item in selected):
        raise ValueError('Each selected feed must be an object.')
    selected_prices = {int(item['rufas_id']): resolve_price(item, library) for item in selected if item.get('available', True)}
    candidate_currencies = {record['currency'] for record in selected_prices.values() if record is not None}
    if len(candidate_currencies) > 1:
        raise ValueError('Candidate feed prices use incompatible currencies.')
    feeds = sorted((build_feed(item, library) for item in selected if item.get('available', True)),
                   key=lambda feed: feed.rufas_id)
    if len(feeds) < 2 or len({feed.rufas_id for feed in feeds}) != len(feeds):
        raise ValueError('Select at least two distinct available feeds.')

    candidate_seed = None
    if candidate_ndf_mode:
        midpoint_amounts = [(feed.lower_limit + feed.limit) / 2 for feed in feeds]
        if sum(midpoint_amounts) <= 0:
            return {'ok': False, 'status': 'incomplete_input', 'message': 'Available feed bounds do not allow a positive candidate ration.'}
        midpoint_feeds = [FeedInRation(amount=amount, info=feed) for amount, feed in zip(midpoint_amounts, feeds)]
        initial_ndf = CandidateRationOptimizer._candidate_ndf(midpoint_feeds)
        if not math.isfinite(initial_ndf):
            raise ValueError('RuFaS feed composition cannot determine candidate-ration NDF.')
        initial_animal = calculate(_payload_with_ndf(payload, initial_ndf), include_native=True)
        if initial_animal['status'] != 'success':
            return {'ok': False, 'status': initial_animal['status'],
                    'error_code': initial_animal.get('error_code', initial_animal['status']),
                    'missing_inputs': initial_animal.get('missing_inputs', []),
                    'message': initial_animal.get('message', 'Complete the animal profile.')}
        initial_requirements = initial_animal['_native_requirements']
        candidate_seed = _allocate_candidate_seed(feeds, initial_requirements.dry_matter)
        seed_feeds = [FeedInRation(amount=amount, info=feed) for amount, feed in zip(candidate_seed, feeds)]
        seed_ndf = CandidateRationOptimizer._candidate_ndf(seed_feeds)
        if not math.isfinite(seed_ndf):
            raise ValueError('RuFaS feed composition cannot determine candidate-ration NDF.')
        payload = _payload_with_ndf(payload, seed_ndf)

    animal = calculate(payload, include_native=True)
    if animal['status'] != 'success':
        return {'ok': False, 'status': animal['status'],
                'error_code': animal.get('error_code', animal['status']),
                'missing_inputs': animal.get('missing_inputs', []),
                'message': animal.get('message', 'Complete the animal profile.')}
    requirements = animal.pop('_native_requirements')
    if candidate_ndf_mode and not math.isclose(
            requirements.dry_matter, initial_requirements.dry_matter, rel_tol=0, abs_tol=1e-8):
        return {'ok': False, 'status': 'calculation_error',
                'message': 'Candidate NDF changed the lactating-cow DMI requirement; candidate formulation is not validated.'}

    emissions = explicit_emissions
    methane = emissions['enteric_methane_g_day'] if emissions else None
    urine_nitrogen = emissions['urine_nitrogen_kg_day'] if emissions else None
    if candidate_ndf_mode or emissions is None:
        optimizer = CandidateRationOptimizer(
            animal_payload=animal,
            calculate_candidate_emissions=emissions is None,
            seed=candidate_seed,
            candidate_ndf=candidate_ndf_mode,
            calculation_type=calculation_type,
        )
    else:
        optimizer = DeterministicRationOptimizer()

    solution, configuration = optimizer.attempt_optimization(
        nutrient_standard=NutrientStandard.NASEM,
        pen_average_body_weight=animal['body_weight_kg'],
        pen_average_enteric_methane=methane, pen_average_urine_nitrogen=urine_nitrogen,
        requirements=requirements,
        initial_dry_matter_requirement=requirements.dry_matter,
        initial_protein_requirement=requirements.metabolizable_protein,
        pen_available_feeds=feeds, animal_combination=group,
    )
    if not solution.success:
        return {'ok': False, 'status': 'solver_failure', 'solver_status': int(solution.status),
                'message': 'Native RuFaS could not find a feasible ration.'}
    constraints = []
    candidate_constraint_names = {
        '_candidate_net_energy_constraint': 'NASEM_net_energy_constraint',
        '_candidate_protein_lower': 'protein_constraint_lower',
        '_candidate_protein_upper': 'protein_constraint_upper',
    }
    for constraint in optimizer._select_constraints(group):
        constraint_function = constraint['fun']
        slack = float(constraint_function(solution.x, configuration))
        name = candidate_constraint_names.get(constraint_function.__name__, constraint_function.__name__)
        constraints.append({'name': name, 'slack': slack,
                            'tolerance': 1e-4, 'status': 'satisfied' if math.isfinite(slack) and slack >= -1e-4 else 'violated',
                            'satisfied': math.isfinite(slack) and slack >= -1e-4})
    ordered = sorted(feeds, key=lambda feed: feed.rufas_id)
    quantities = [float(value) for value in solution.x]
    if any(not math.isfinite(value) or value < -1e-6 for value in quantities) or not all(c['satisfied'] for c in constraints):
        return {'ok': False, 'status': 'constraint_violation', 'message': 'Native result failed independent constraint checks.', 'constraints': constraints}
    if any(quantity < feed.lower_limit - 1e-4 or quantity > feed.limit + 1e-4
           for quantity, feed in zip(quantities, ordered)):
        return {'ok': False, 'status': 'feed_bound_violation',
                'message': 'Native result violates feed inclusion bounds.'}
    cost = sum(quantity * feed.purchase_cost for quantity, feed in zip(quantities, ordered))
    if not math.isfinite(cost) or not math.isclose(cost, float(solution.fun), rel_tol=1e-5, abs_tol=1e-4):
        return {'ok': False, 'status': 'cost_mismatch',
                'message': 'Native solver cost does not match the feed quantities.'}

    # Recompute the candidate profile at the exact optimized NDF before reporting requirements.
    ration_feeds = [FeedInRation(amount=quantity, info=feed) for quantity, feed in zip(quantities, ordered)]
    ration_context = None
    if candidate_ndf_mode:
        final_ndf = CandidateRationOptimizer._candidate_ndf(ration_feeds)
        if not math.isfinite(final_ndf):
            return {'ok': False, 'status': 'calculation_error', 'message': 'Optimized ration NDF is not finite.'}
        final_animal = calculate(_payload_with_ndf(payload, final_ndf), include_native=True)
        if final_animal['status'] != 'success':
            return {'ok': False, 'status': final_animal['status'],
                    'missing_inputs': final_animal.get('missing_inputs', []),
                    'message': final_animal.get('message', 'Could not calculate final candidate requirements.')}
        final_requirements = final_animal['_native_requirements']
        if (not math.isclose(final_requirements.dry_matter, requirements.dry_matter, rel_tol=0, abs_tol=1e-8)
                or not math.isclose(final_requirements.total_energy_requirement,
                                    requirements.total_energy_requirement, rel_tol=0, abs_tol=1e-8)):
            return {'ok': False, 'status': 'calculation_error',
                    'message': 'Candidate NDF changes DMI or energy requirements; candidate formulation is not validated.'}
        animal = final_animal
        requirements = final_requirements
        ration_context = {'ndf_percentage': final_ndf,
                          'dry_matter_intake_kg_day': sum(quantities),
                          'source': 'RuFaS optimized feed composition'}
    elif has_previous_ration:
        ration_context = {'ndf_percentage': animal['parameters']['ndf_percentage'],
                          'source': 'RuFaS previous-ration feed composition'}
    _enrich_constraints(constraints, requirements, configuration)
    if emissions is None:
        emissions = calculate_candidate_ration_emissions(ration_feeds, animal, calculation_type)
    elif candidate_ndf_mode:
        emissions = {**emissions,
                     'candidate_ration': [{'rufas_id': feed.info.rufas_id, 'kg_dm_day': feed.amount}
                                          for feed in ration_feeds],
                     'candidate_ration_dmi_kg_day': sum(quantities)}

    # The allowed milk is a model-derived ceiling from the same final ration,
    # requirements and energy losses used by the optimizer. It only applies to
    # lactating cows; the same diet must keep the current production reference.
    milk_potential = None
    if is_lactating:
        milk_potential = evaluate_milk_potential(
            ration_feeds, animal, requirements, emissions, reference_milk)
        if milk_potential['status'] == 'unavailable':
            return {'ok': False, 'status': 'calculation_error',
                    'error_code': 'milk_potential_unavailable',
                    'message': 'The native milk potential could not be calculated for this ration.',
                    'milk_potential': milk_potential}
        if not allow_below_reference and milk_potential['allowed_kg_day'] < reference_milk - FLOOR_TOLERANCE_KG_DAY:
            return {'ok': False, 'status': 'constraint_violation',
                    'error_code': 'milk_floor_violation',
                    'message': 'The lowest-cost ration would allow less milk than the current production reference.',
                    'milk_potential': milk_potential}

    alternative_solutions = []
    if explore_alternatives and is_lactating:
        try:
            samples = int(payload.get('alternative_samples_per_target', 20))
        except (TypeError, ValueError):
            samples = 20
        samples = max(0, min(samples, 200))
        if samples:
            from services.diet_exploration import generate_alternatives
            target_raw = parameters.get('milk_production')
            target_value = (
                parse_number(target_raw, 'milk_production', minimum=0)
                if target_raw is not None and not (isinstance(target_raw, str) and not target_raw.strip())
                else reference_milk
            )
            try:
                alternative_solutions = generate_alternatives(
                    quantities, ordered, optimizer, configuration, group, animal,
                    requirements, emissions, reference_milk, calculation_type,
                    target_value, samples, seed=int(round(target_value * 1000)),
                )
            except (ArithmeticError, TypeError, ValueError):
                alternative_solutions = []

    ingredients = [{'id': feed.rufas_id, 'name': feed.Name, 'kg_dm': quantity,
                    'kg_as_fed': quantity / (feed.DM / 100),
                    'price_as_fed': feed.purchase_cost * feed.DM / 100,
                    'cost_day': quantity * feed.purchase_cost,
                    'percent_dm': 100 * quantity / sum(quantities),
                    'percent_cost': 100 * quantity * feed.purchase_cost / cost if cost > 0 else None,
                    'price_record': selected_prices[feed.rufas_id]}
                   for quantity, feed in zip(quantities, ordered)
                   # In automatic mode the pool has many candidates; report only those
                   # actually included in the ration (non-zero dry matter).
                   if selection_mode != 'automatic' or quantity > 1e-6]
    fixed_inclusions = [
        {'id': feed.rufas_id, 'name': feed.Name, 'kg_dm': feed.lower_limit}
        for feed in ordered if abs(feed.limit - feed.lower_limit) <= 1e-9
    ]
    # These are direct sums of native feed composition, not inferred NASEM energy or MP supply.
    nutrient_supply = {'dry_matter': {'value': sum(quantities), 'unit': 'kg/day', 'source': 'RuFaS optimized quantities'}}
    for label, field_name in (('ndf', 'NDF'), ('crude_protein', 'CP'), ('starch', 'starch'),
                              ('fat', 'EE'), ('calcium', 'calcium'), ('phosphorus', 'phosphorus')):
        value = NutritionSupplyCalculator._calculate_nutritive_content(ration_feeds, field_name)
        if math.isfinite(value):
            nutrient_supply[label] = {'value': value, 'unit': 'kg/day', 'source': 'RuFaS NASEM feed composition'}
    baseline = {'status': 'not_available', 'source': payload.get('previous_ration_source'), 'missing_price_ids': [], 'price_records': []}
    if payload.get('previous_ration') is not None:
        baseline['status'] = 'complete'
        baseline_cost = 0.0
        for item in payload['previous_ration']:
            identifier = int(item['rufas_id'])
            try:
                price_record = resolve_price(item, library) if item.get('price_record') is not None or item.get('price_as_fed') not in (None, '') else selected_prices.get(identifier)
            except ValueError:
                # The optional historical quote cannot determine today's nutrient supply.
                price_record = None
            if price_record is None:
                baseline['missing_price_ids'].append(identifier)
                continue
            baseline['price_records'].append(price_record)
            price = price_record['price_as_fed']
            dm_fraction = float(library[identifier]['DM']) / 100
            baseline_cost += parse_number(item['kg_dm_day'], 'kg_dm_day', minimum=0) / dm_fraction * price
        if baseline['missing_price_ids']:
            baseline['status'] = 'missing_prices'
        else:
            basis = comparison_status(baseline['price_records'] + list(selected_prices.values()))
            baseline['comparability'] = basis
            baseline.update({'cost_day': baseline_cost, 'cost_month': baseline_cost * 30})
            if basis == 'comparable':
                savings = baseline_cost - cost
                baseline.update({'daily_savings': savings, 'monthly_savings': savings * 30,
                                 'percentage_savings': savings / baseline_cost * 100 if baseline_cost > 0 else None})
    result = {'ok': True, 'status': 'success', 'engine': 'RuFaS RationOptimizer',
              'nutritional_standard': 'NASEM_2021', 'selection_mode': selection_mode,
              'animal_category': category, 'stage_type': calculation_type,
              'lactating': is_lactating,
              'cost_day': cost, 'cost_month': cost * 30,
              'currency': next(iter(candidate_currencies)),
              'dmi': sum(quantities), 'ingredients': ingredients,
              'fixed_inclusions': fixed_inclusions,
              'requirements': animal['requirements'], 'constraints': constraints,
              'requirement_units': {name: item['unit'] for name, item in animal['constraints'].items()},
              'nutrient_supply': nutrient_supply, 'baseline_comparison': baseline,
              'profile': {'animal_category': animal['animal_category'],
                          'source': animal['profile_source'],
                          'assumption_type': animal['assumption_type']},
              'solver_status': int(solution.status), 'emissions': emissions,
              'milk_potential': milk_potential, 'milk_reference_source': reference_source,
              'milk_floor_enforced': is_lactating and not allow_below_reference,
              'alternative_solutions': alternative_solutions,
              'price_unit': 'currency/kg as-fed input; currency/kg dry matter optimizer'}
    if has_previous_ration:
        from services.emissions import previous_ration_feeds
        from services.diet_comparison import compare_rations
        previous = previous_ration_feeds(payload['previous_ration'], library, build_feed)
        result['diet_comparison'] = compare_rations(
            previous, ration_feeds, configuration, selected_prices, payload, animal, emissions,
            calculation_type=calculation_type)
    if ration_context is not None:
        result['ration_context'] = ration_context
    return result
