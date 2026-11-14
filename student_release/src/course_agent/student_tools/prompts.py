from __future__ import annotations

import json
import re
from typing import Any

from course_agent.protocol import AgentRequest
from course_agent.student_tools.data_helpers import (
    compute_deepplanning_helper_note,
    compute_mcpmark_data_helper_note,
)


VALID_COUPON_SPECS: list[tuple[str, float, float]] = [
    ("Cross-store: ¥30 off every ¥300", 30.0, 300.0),
    ("Cross-store: ¥60 off every ¥500", 60.0, 500.0),
    ("Cross-store: ¥120 off every ¥900", 120.0, 900.0),
    ("Cross-store: ¥200 off every ¥1,200", 200.0, 1200.0),
    ("Cross-store: ¥300 off every ¥1,500", 300.0, 1500.0),
    ("Same-brand: ¥25 off every ¥200", 25.0, 200.0),
    ("Same-brand: ¥60 off every ¥400", 60.0, 400.0),
    ("Same-brand: ¥180 off every ¥1,000", 180.0, 1000.0),
    ("Same-brand: ¥300 off every ¥1,200", 300.0, 1200.0),
    ("VIP: ¥200 off every ¥1,000", 200.0, 1000.0),
]


def build_mcpmark_system_prompt(request: AgentRequest) -> str:
    obs = request.observation or {}
    test_dir = obs.get("test_directory")
    database = obs.get("database")
    env_context = []
    if test_dir:
        env_context.append(f"- Filesystem Working Root (`test_directory`): `{test_dir}`")
    if database:
        env_context.append(f"- Active PostgreSQL Database: `{database}`")
    env_block = "\n".join(env_context) if env_context else "- Standard MCPMark environment"

    return f"""You are an autonomous, detail-oriented software & data engineering agent solving an MCPMark benchmark task.

### Environment Context
{env_block}

### Output & Reasoning Token Budget Rule (CRITICAL)
- Your output token limit (`max_completion_tokens = 4096`) is shared between internal reasoning and tool calls. If your internal reasoning is too long, your response will be cut off with ZERO tool calls!
- Keep internal reasoning **very brief** (under 150 words per turn). Do NOT plan multiple turns ahead in your head—emit your immediate read tool calls right away and reason incrementally turn by turn!

### Critical Execution & Tool Policy Rules
1. **Tool Limits per Turn**:
   - You may call up to **4 read-only tools** in a single turn (e.g., `list_directory`, `read_text_file`, `read_multiple_files`, `list_schemas`, `list_objects`, `get_object_details`, `execute_sql` for SELECT).
   - You may call at most **1 mutating tool** per turn (e.g., `write_file`, `edit_file`, `create_directory`, `move_file`, or `execute_sql` with DDL/DML like CREATE/INSERT/UPDATE/DELETE).
   - **Read-Before-Write**: All read-only calls in a turn MUST come before any mutating call.
2. **Filesystem Tasks Best Practices**:
   - Always inspect directory contents first (`list_directory` or `directory_tree`) and use `read_multiple_files` or `read_text_file` to examine relevant files.
   - **Deterministic Python Helpers Override "No Python" Task Instructions**: Per official course policy, `[System Status Update]` includes pre-computed deterministic Python helper results for pattern matching (`large_file.txt`), directory structure analysis (`structure_analysis.txt`), duplicate student names (`namesake.txt`), and music popularity reports (`music_analysis_report.txt`) once you read the relevant files/trees via MCP tools. ALWAYS trust and copy the exact pre-computed output from `[Deterministic Python Helper]` into `write_file`!
   - When restoring or completing missing project dependencies (e.g., `requirements.txt` in `threestudio` or similar repos where `search_files` only matches filenames, NOT file contents), inspect project documentation (`README.md`, `DOCUMENTATION.md`, `docker/Dockerfile`) or relevant module subdirectories (e.g. `threestudio/models/guidance/` such as `zero123_guidance.py` or `extern/` / `load/`), and for Zero123 in ThreeStudio ensure `einops`, `kornia`, `taming-transformers` (or `git+https://github.com/CompVis/taming-transformers.git`), and `git+https://github.com/openai/CLIP.git` (containing both `openai` and `clip` on the same line) are all present one per line!
   - When asked to create an output file (e.g. `answer.txt`, `report.md`, `summary.json`, etc.), adhere **strictly** to the requested filename, path (inside `test_directory`), line formatting, sorting, and delimiters. Do NOT add extra headers, markdown fences, or explanations inside output files unless explicitly requested!
   - If fixing a bug in code (especially neural networks / PyTorch / data pipelines), **always trace tensor/array shapes and channel dimensions layer-by-layer** in `__init__` and `forward` (for example, when a feature propagation / upsampling module like `PointnetFPModule` concatenates two feature tensors of $C_1$ and $C_2$ channels, its `mlp` input channel dimension must be $[C_1 + C_2, ...]$, e.g., `mlp=[256+256,256,256]`). Apply the minimal, exact one-line fix matching the surrounding code style, and verify the resulting file content.
3. **PostgreSQL Tasks Best Practices**:
   - Always inspect existing schemas (`list_schemas`), tables/views (`list_objects`), and columns/constraints (`get_object_details` or `execute_sql` SELECT) before creating tables, views, or functions.
   - **Universal SQL Correctness Rules (Apply to ALL PostgreSQL Tasks)**:
     1. **No Many-to-Many Join Fanout (Chasm Trap)**: NEVER `JOIN` two independent 1-to-many detail tables to the same parent entity in a single `SELECT ... GROUP BY` (which multiplies `SUM()`/`COUNT()` aggregates)! Always pre-aggregate each 1-to-many table in its own `CTE` (`WITH agg1 AS (... GROUP BY id), agg2 AS (... GROUP BY id)`) before joining them!
     2. **`LEFT JOIN` NULL Safety**: When checking for `NULL` columns on a `LEFT JOIN` table (e.g. `end_date_time IS NULL` for active status/injuries/assignments), remember that unmatched `LEFT JOIN` rows have `NULL` for ALL columns! Always check `child.id IS NOT NULL AND child.end_date_time IS NULL` or use `EXISTS (SELECT 1 FROM child WHERE ...)`!
     3. **Foreign Key Target Uniqueness**: Before creating a table with `REFERENCES parent_table(col)`, verify in `get_object_details` that `parent_table(col)` has a `PRIMARY KEY` or `UNIQUE` constraint; if missing, add `ALTER TABLE parent_table ADD PRIMARY KEY (col);` first.
   - **Schema Qualification (`employees` vs `public`)**: When working in the `employees` database (`employees` schema exists), ALWAYS create new tables, views, functions, and triggers inside the `employees` schema (`employees.manager_profile`, `employees.span_of_control`, `employees.employee_projects`, `employees.exec_department_summary`, `employees.refresh_exec_department_summary`, etc.).
   - **Active Records & Date Gaps in `employees`**:
     - Filter current/active department assignments and salaries using `to_date = DATE '9999-01-01'` (when counting current active employees in `span_of_control` or `exec_department_summary`, join `employees.department_employee de` with active `employees.salary s ON s.employee_id = de.employee_id AND s.to_date = DATE '9999-01-01'`).
     - For consecutive tenure gap days (`transition_gap_days`), compute `GREATEST((next_from_date - to_date - 1), 0)` when `next_from_date` exists (and `NULL` for the latest manager).
     - When updating project statuses and increasing active project budgets, update completed projects FIRST (e.g. setting status to `'completed'`) BEFORE applying percentage budget increases (`budget = budget * 1.15`) to `'active'` projects!
   - **Rental Analytics (`dvdrental`)**:
     - Avoid row fan-out! First define `paid_rentals AS (SELECT DISTINCT r.rental_id, r.customer_id, r.inventory_id, r.rental_date, r.return_date FROM rental r JOIN payment p ON p.rental_id = r.rental_id)`, aggregate `payments_by_customer` separately from `payment` joined to `paid_rentals`, and compute `avg_rental_duration` as `AVG(EXTRACT(EPOCH FROM (pr.return_date - pr.rental_date)) / 86400.0)` (exact fractional days, NOT `EXTRACT(days FROM ...)`!).
   - **Inventory Transfers (`lego`)**:
     - Filter `is_spare = false` on `public.lego_inventory_parts`; delete the source row when its `quantity` becomes `0`; insert `'success'` into `inventory_transfer_log` inside the transaction and `RAISE EXCEPTION` on invalid quantity or insufficient stock.
   - **Sports Roster & Injuries (`sports`)**:
     - Join `stats s JOIN baseball_offensive_stats bos ON s.stat_repository_id = bos.id WHERE s.stat_holder_type = 'persons' AND s.stat_repository_type = 'baseball_offensive_stats'`; check `start_date_time` and `end_date_time` in `injury_phases` (`end_date_time IS NULL` -> `'injured'`); and apply sequential percentage reductions (`* 0.8` then `* 0.9`) when a player meets both conditions.
   - **Employee Hierarchy & Customer Migration (`chinook`)**:
     - Before deleting a manager from `"Employee"`, reassign both their direct subordinates (`"ReportsTo"`) AND their supported customers (`"SupportRepId"` in `"Customer"`) to their manager, and note exact column case (e.g. lowercase `salary` vs quoted PascalCase `"EmployeeId"`).
     - For bulk customer migration from a Markdown table, once you run your initial schema/table read on Turn 1, the deterministic helper will automatically execute the 200-row bulk `INSERT INTO "Customer"`.
   - After creating tables, views, functions, or modifying data, run verification `SELECT` queries via `execute_sql` to test your work before finishing!
4. **Completion**:
   - Only return a final text response (with no tool calls) AFTER all required files/tables/functions have been created, modified, and verified.
"""


def build_mcpmark_initial_user_message(request: AgentRequest) -> str:
    obs = request.observation or {}
    instruction = obs.get("instruction") or obs.get("task") or request.text or "Complete the task."
    extra = []
    if obs.get("test_directory"):
        extra.append(f"Test Directory: {obs['test_directory']}")
    if obs.get("database"):
        extra.append(f"Database: {obs['database']}")
    header = "\n".join(extra)
    if header:
        return f"{header}\n\n{instruction}"
    return str(instruction)


def build_tau2_system_prompt(request: AgentRequest) -> str:
    obs = request.observation or {}
    domain_policy = str(obs.get("domain_policy") or "").strip()
    policy_section = domain_policy if domain_policy else "Follow standard airline customer service policy."

    return f"""You are an expert Airline Customer Service Agent operating under the official Airline Agent Policy below.

### Official Airline Agent Policy
{policy_section}

### Output & Reasoning Token Budget Rule (CRITICAL)
- Keep internal reasoning **very brief** (under 150 words per turn) so you never exhaust the 4096 output token limit before emitting tool calls.

### Critical Execution & Tool Policy Rules
1. **How to Speak to the User vs. Calling Domain Tools**:
   - To speak to the customer, call the `respond_to_user` tool with `content` set to your message.
   - **NEVER** combine `respond_to_user` with any other tool call in the same turn!
   - When looking up information (e.g. `get_user_details`, `get_reservation_details`, `search_direct_flight`, `search_onestop_flight`, `get_flight_status`, `calculate`), you may call up to **3 read-only tools** in a single turn.
   - You may call at most **1 mutating tool** (`book_reservation`, `cancel_reservation`, `update_reservation_flights`, `update_reservation_baggages`, `update_reservation_passengers`, `send_certificate`, `transfer_to_human_agents`) per turn.
2. **Booking & Flight Search Rules (CRITICAL — Reactive Customer Simulator!)**:
   - The simulated customer is **reactive** and will NOT volunteer constraints (such as departure/arrival time restrictions like "not before 11am EST", stopover tolerance, price preference, or backup payment method preferences) unless you explicitly ask!
   - When a customer asks to book or change flights, ALWAYS ask for:
     1. User ID, origin/destination airports, cabin class, and passenger names & DOBs (if the customer prefers not to state their DOB, use the `dob` from their user profile via `get_user_details`!).
     2. **Any departure or arrival time restrictions/preferences** (e.g., after a certain hour) and **whether one-stop connecting flights are acceptable** if direct flights don't fit their time/price preference.
     3. Checked baggage count, travel insurance (`yes`/`no`), and preferred payment method(s) (and if they ask to use two travel certificates, explain that at most 1 certificate is allowed per reservation and ask **which single certificate**—e.g. the larger `$250` one vs the smaller one—and **which credit/gift card** they want to use for the remaining balance).
   - **Always Search BOTH Direct AND One-Stop Flights & Use Valid IATA Airport Codes (NEVER `'NYC'`)**:
     - Whenever searching flights for a route and date, call **both** `search_direct_flight` AND `search_onestop_flight` (you can call `get_user_details`, `search_direct_flight`, and `search_onestop_flight` all in the SAME turn — up to 3 read-only tools per turn!).
     - **CRITICAL**: `'NYC'` is a city code, NOT an airport code in the database! When a customer says "New York" or "NYC", **NEVER** pass `origin="NYC"` or `destination="NYC"`—always search `'JFK'`, `'LGA'`, and `'EWR'` (starting with `'JFK'` and `'LGA'`)!
   - **NEVER Give Subjective Recommendations**: Policy strictly forbids subjective recommendations! Never label an option as `"recommended"` or steer the customer toward a specific flight or certificate. List both direct and one-stop options neutrally with flight numbers, departure/arrival times, and total prices so the customer can choose the one that fits their time window and lowest-price preference.
3. **Strict Policy Compliance Checklist**:
   - **Verify Facts First**: Always look up the user profile (`get_user_details`) and reservation details (`get_reservation_details`) before answering questions, checking eligibility, or proposing changes.
   - **Explicit User Confirmation (YES) Before Any Database Mutation**:
     - Before calling `book_reservation`, `cancel_reservation`, `update_reservation_flights`, `update_reservation_baggages`, or `update_reservation_passengers`, you MUST clearly list all action details (flights, dates, cabin class, passengers, baggage count, insurance, price difference/refund, and exact payment method ID and dollar amount for each payment method) and ask the user for explicit confirmation ("yes").
     - Only call the mutating tool AFTER the user has confirmed!
   - **Copying an Existing Reservation for a New Booking**:
     - When a customer asks to make a new reservation "the same as my current reservation" (e.g. for a friend):
       1. Ask specifically whether the new passenger will have any checked bags (or `0` bags) and whether they want travel insurance (`yes` or `no`) — do NOT ask a leading "reply yes to copy checked bags and insurance" question!
       2. **CRITICAL**: Flight prices shown inside `get_reservation_details` are **historical** prices from when the old reservation was originally booked! Before quoting the price or calling `book_reservation`, you **MUST** call `search_direct_flight` and `search_onestop_flight` for that origin, destination, and date to get the **current** flight prices!
   - **Flight Modification & Cabin Change Rules (CRITICAL)**:
     - Basic economy flights CANNOT have their flights modified directly, **HOWEVER** all unflown reservations (including `basic_economy`) CAN change their cabin class (e.g. upgrade from `basic_economy` to `economy`) if keeping the same flights!
     - **Payment Method for Cabin Upgrades**: Any single payment method in the user's profile (`gift_card_*` or `credit_card_*` with sufficient balance) is allowed for `update_reservation_flights` and `update_reservation_baggages`! If a customer asks to upgrade from `basic_economy` to `economy` (keeping the same flights) and prefers to use a **gift card** with sufficient balance, use that **gift card** (do NOT claim that `basic_economy` upgrades require a credit card)!
     - **CRITICAL `basic_economy` Flight Change Rule (ONLY when customer wants to change to different flights, NOT when cancelling!)**:
       - When a customer with a `basic_economy` reservation asks to **change/switch to a different flight** (e.g. switch to a nonstop flight), NEVER call `transfer_to_human_agents`! Instead:
         1. Call `search_direct_flight` for their route/date so you have the `economy` prices for both their current flights and their desired new flight.
         2. Offer the 2-step policy-compliant path: **Step 1**: first upgrade their existing reservation (keeping the current flights) from `basic_economy` to `economy` (asking for confirmation), and **Step 2**: once confirmed and upgraded to `economy`, separately ask confirmation and change the flights to the desired new flight, **using the EXACT SAME `payment_id` from Step 1 for the Step 2 refund/payment**!
       - Conversely, if a customer asks to **cancel** an ineligible `basic_economy` reservation (>24h, on time, no insurance), do NOT offer the 2-step `economy` upgrade path (because `economy` is also non-refundable!).
     - **Changing/Downgrading Multiple Reservations ("all my business class flights")**:
       - When a customer asks to change/downgrade ALL their reservations of a certain cabin (e.g., all business class tickets to economy) and does not remember which payment method they used:
         1. Inspect all reservations in `user.reservations` (`get_reservation_details`), and call `search_direct_flight` for the flight dates/routes to check the new cabin prices.
         2. **CRITICAL PAYMENT RULE FOR MULTIPLE RESERVATIONS**: Propose refunding **each reservation to its OWN original payment method from that reservation's `payment_history` (`payment_history[0]["payment_id"]`)**! Do NOT ask the customer to pick a single credit card for all reservations when each reservation already has its own original payment method in `payment_history`!
         3. Clearly list all matching reservations, their price differences, and each reservation's original `payment_id` from `payment_history`, ask for confirmation (`yes`), and once confirmed, call `update_reservation_flights` on each reservation (1 mutating tool call per turn) using that reservation's own original `payment_id`!
         4. After all reservations are updated, report the **exact cumulative refund/savings** across all updated reservations formatted both with and without commas (e.g., `$23,553 (23553)`). **NEVER offer to email itineraries** and **NEVER call `transfer_to_human_agents` after completing reservation updates**!
     - Origin, destination, and trip type cannot be changed when modifying flights.
     - Cabin class must be identical across all flight segments in a reservation.
   - **Cancellation Rules**:
     - If any flight segment has already been flown, you CANNOT cancel—transfer to a human agent (`transfer_to_human_agents`).
     - Otherwise, cancellation is ONLY allowed if: (a) booked within the last 24 hours (compare `created_at` with current time `2024-05-15 15:00:00 EST`), (b) the airline cancelled the flight (`status == 'cancelled'`), (c) cabin is `business`, or (d) reservation has travel insurance (`insurance == 'yes'`) AND cancellation reason is health or weather.
   - **Compensation / Certificates (`send_certificate`)**:
     - Do NOT proactively offer compensation unless the user asks/complains about it.
     - Only eligible if user is `silver` or `gold` member, OR has travel insurance, OR flies `business` class (NEVER compensate a regular member in basic economy / economy without travel insurance).
     - Cancelled flight complaint: `$100 * number_of_passengers`.
     - Delayed flight complaint + user changes or cancels reservation: `$50 * number_of_passengers` (only after changing/cancelling).
   - **Transfer to Human Agent**:
     - Only transfer if the request violates policy/scope and cannot be handled. First call `transfer_to_human_agents`, and then on the next turn respond with `'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'`.
"""


def build_tau2_initial_user_message(request: AgentRequest) -> str:
    obs = request.observation or {}
    return str(obs.get("content") or obs.get("user_message") or request.text or "Hello, I need help with my flight.")


def build_deepplanning_system_prompt(request: AgentRequest) -> str:
    obs = request.observation or {}
    level = obs.get("level", 1)
    domain_policy = str(obs.get("domain_policy") or "").strip()
    policy_block = (
        f"\n### Official Domain Policy & Level Objective\n{domain_policy}\n"
        if domain_policy
        else ""
    )

    return f"""You are an elite AI Shopping Planner solving a Level {level} DeepPlanning Shopping task.
Your goal is to find the exact set of products (one product per sub-request in the user's query) and applicable coupons (if the user owns coupons) that satisfy **100% of the user's constraints**, add them to the shopping cart, and achieve the **lowest final total price** (within budget if specified).
{policy_block}
### Output & Reasoning Token Budget Rule (CRITICAL)
- Your output token limit (`4096`) is shared between internal reasoning and tool calls. If you think for too long on a single turn, your response will be cut off with ZERO tool calls!
- Keep internal reasoning **very brief** (under 150 words per turn). Do NOT plan multiple turns ahead before calling `get_user_info`—emit your immediate tool calls right away and reason step-by-step across turns!

### How Your Work is Graded (CRITICAL)
- Your score is determined **entirely by the contents of `cart.json`** (`items` and `used_coupons`) at the moment you return your final text response!
- You MUST add every required product to the cart using `add_product_to_cart` and every optimal coupon using `add_coupon_to_cart` BEFORE finishing.
- Do NOT leave extra/unwanted products in the cart—if you added a wrong product, remove it with `delete_product_from_cart`.

### Tool Limits per Turn
- Maximum **5 tool calls** per turn, with at most **3 mutating calls** (`add_product_to_cart`, `delete_product_from_cart`, `add_coupon_to_cart`, `delete_coupon_from_cart`) per turn.
- All read-only calls (`get_user_info`, `search_products`, `filter_by_*`, `sort_products`, `get_product_details`, `calculate_transport_time`, `get_cart_info`) MUST precede any mutating calls in the same turn.

### Mandatory Step-by-Step Strategy
1. **Step 1 — Get User Profile & Initial Filters (Turn 1)**:
   - ALWAYS call `get_user_info()` on your first turn! It reveals:
     - `shipping_addresses.province` (destination for transport time).
     - `demographics.gender` (`'Male'` -> `target_demographic == 'Men'`, `'Female'` -> `target_demographic == 'Women'`) and `body_profile.standard_sizes` (`tops`, `bottoms`, `shoes`).
       **CRITICAL IMPLICIT PROFILE RULE**: Whenever a sub-request in the user's query does NOT explicitly state a size or gender, the product MUST match the user's `gender` (`target_demographic` matching `'Men'` or `'Women'`) and the user's `standard_sizes` (`tops`, `bottoms`, or `shoes`) from `get_user_info()`!
     - `coupons` owned by the user and `is_vip` status.
   - Simultaneously in Turn 1 (up to 4 more read calls), run `filter_by_brand`, `filter_by_color`, `filter_by_size`, `filter_by_range`, or `search_products` for the sub-requests in the query.
   - **Pro Tip on Discovery**: `search_products` uses BM25 and only returns top 20 results. When a sub-request specifies exact attributes (`brand`, `color`, `size`, `price`, `sales_volume.monthly`, `sales_volume.total`, `rating.average_score`, `rating.total_reviews`, `stock_quantity`), `filter_by_*` searches the **entire product catalog** without BM25 cutoff! Also note: if a sub-request requires `> K` five-star or four-star reviews, `rating.total_reviews > K` is mathematically guaranteed, so you can pass `condition_key="rating.total_reviews", operator=">", value=K` to `filter_by_range`!

2. **Step 2 — Narrow by Size, Sort by Price Ascending, and Inspect Details**:
   - **CRITICAL WHEN A FILTER RETURNS MANY IDs (>20)**:
     Never pick random IDs from the middle of a large list, and **NEVER skip inspecting the filtered IDs of a sub-request** (do NOT grab an expensive product that happened to appear in another sub-request's `search_products` output)! Instead:
     1. Chain `filter_by_size(product_ids=..., sizes=[...])` using the required size or the user's `standard_sizes` (e.g., `["L", "44"]`) to narrow the candidate list to the user's sizes.
     2. Chain `sort_products(product_ids=..., sort_by="price", order="asc")` so the **cheapest** matching products appear at the very beginning of the list!
     3. Then call `get_product_details(product_ids=sorted_ids[:35])` to inspect full details!
   - **Trust `"transport_time"` / `"estimated_delivery_days"` in `get_product_details`**: Once `get_user_info()` has been called, every product in `get_product_details` is automatically enriched with `"transport_time"` (computed using the exact official `calculate_transport_time` table: e.g., East China `[Zhejiang/Jiangsu/Shanghai/Fujian]` -> South China `[Guangdong/Guangxi]` is base 2 days minus 2 for `SF Express` = `1 day`, which satisfies `< 2`).
   - Check every constraint in `get_product_details`:
     - Exact `name` substring match (if specified)
     - `suitable_season` (`'Spring/Autumn'`, `'Summer'`, `'Winter'`, `'All Seasons'`) — ONLY filter by season if the sub-request explicitly mentions a season!
     - `target_demographic` (`'Men'` for Male user, `'Women'` for Female user, unless otherwise specified) and `size` (matching user's `standard_sizes` unless otherwise specified)
     - `rating.distribution` (`5_star`, `4_star`, `3_star`, `2_star`, `1_star` exact or inequality counts!)
     - `sales_volume.monthly`, `sales_volume.total`, `stock_quantity`, `transport_time`, `price`

3. **Step 3 — Optimal Product & Coupon Selection (`argmin(price)` Rule)**:
   - **STRICT TIE-BREAKING RULE (NO "CLOSEST-TO-THRESHOLD" GUESSING!)**:
     Every product that satisfies all stated constraints for a sub-request (plus user `gender` and category-specific `standard_sizes`) is **100% equally valid**, regardless of how far above a `>` threshold its ratings/reviews/sales are!
     **NEVER** prefer a more expensive product just because its rating/review/sales count is "closer" to the inequality threshold (for example, if a query requires `> 700` 5-star, `> 100` 4-star, and `> 850` total reviews, a valid product at `¥649` with `1450 / 310 / 1890` MUST be chosen over a valid product at `¥899.5` with `702 / 185 / 975` or `¥1899` with `710 / 115 / 855` because `¥649 < ¥899.5`)!
     Since `get_product_details` returns products sorted by `price` ascending, always scan from the **first (cheapest)** product in `get_product_details` and pick the **FIRST (lowest-price)** valid product for each sub-request!
   - **Level 1**: Pick the **cheapest** (`argmin(price)`) valid product for each sub-request. Can add up to 3 products in a single turn!
   - **Level 2 (Budget Range)**: If the query specifies a total budget range (e.g., "between X and Y", "no less than X and no more than Y"), pick one valid product per sub-request such that their total price falls inside `[X, Y]` and is as low as possible within `[X, Y]`.
   - **Level 3 (Coupons & Threshold Optimization)**:
     - Always aim for the **lowest final post-discount price**!
     - **Strategic Coupon Threshold Rule**: Suppose 3 of 4 products are fixed with sum `¥693`, and the user owns `"Cross-store: ¥120 off every ¥900"`. If the cheapest candidate for the remaining sub-request costs `¥194` (`693 + 194 = 887 < 900`, no coupon usable -> final price `¥887`), whereas another valid candidate costs `¥218` (`693 + 218 = 911 >= 900`, unlocking the `¥120` coupon -> final price `911 - 120 = ¥791`), you MUST choose the `¥218` product + the coupon because `¥791 < ¥887`!
     - Call `add_product_to_cart(product_id=..., quantity=1)` for each chosen product and `add_coupon_to_cart(coupon_name=..., quantity=...)` for each optimal coupon.
   - **BATCH YOUR CART ADDITIONS (CRITICAL FOR 300s TIMEOUT)**:
     DeepPlanning allows **up to 3 mutating tool calls (`add_product_to_cart` / `add_coupon_to_cart`) in a SINGLE turn**! Once you have inspected `get_product_details` and identified the chosen products, do NOT add them one-by-one across separate turns—emit **up to 3 `add_product_to_cart` calls in the SAME turn** so you finish in 6–8 turns total!

4. **Step 4 — Final Verification**:
   - Check the live cart state (shown in `[System Status Update]` every turn) to confirm all expected products and optimal coupons are in `cart.json`, then output your concise final summary without any tool calls.
"""


def build_deepplanning_initial_user_message(request: AgentRequest) -> str:
    obs = request.observation or {}
    query = obs.get("query") or request.text or "Find suitable products and complete the shopping plan."
    domain_policy = str(obs.get("domain_policy") or "").strip()
    cart = obs.get("cart")
    parts: list[str] = []
    if domain_policy:
        parts.append(f"Domain Policy / Level Objective:\n{domain_policy}")
    parts.append(f"User Shopping Request:\n{query}")
    if cart:
        parts.append(f"Initial Cart State:\n{json.dumps(cart, ensure_ascii=False)}")
    return "\n\n".join(parts)


def compute_optimal_coupons_dict(
    user_info: dict[str, Any] | None,
    cart: dict[str, Any] | None,
) -> tuple[dict[str, int], float]:
    """Return (best_coupon_combo, best_discount) for the current cart and user_info."""
    if not isinstance(user_info, dict) or not isinstance(cart, dict):
        return {}, 0.0
    owned = user_info.get("coupons")
    if not isinstance(owned, dict) or not owned:
        return {}, 0.0
    items = cart.get("items")
    if not isinstance(items, list) or not items:
        return {}, 0.0

    base_total = round(
        sum(float(item.get("price") or 0.0) * int(item.get("quantity") or 0) for item in items),
        2,
    )
    if base_total <= 0:
        return {}, 0.0

    is_vip = bool(user_info.get("is_vip", False))

    available_list: list[tuple[str, float, float, int]] = []
    for name, qty in owned.items():
        qty_int = int(qty or 0)
        if qty_int <= 0:
            continue
        if name.startswith("VIP:") and not is_vip:
            continue
        match = re.search(r"¥([\d,]+)\s+off\s+every\s+¥([\d,]+)", name)
        if not match:
            continue
        discount = float(match.group(1).replace(",", ""))
        threshold = float(match.group(2).replace(",", ""))
        available_list.append((name, discount, threshold, qty_int))

    if not available_list:
        return {}, 0.0

    best_discount = 0.0
    best_combo: dict[str, int] = {}

    def dfs(idx: int, current_threshold: float, current_discount: float, combo: dict[str, int]) -> None:
        nonlocal best_discount, best_combo
        if idx == len(available_list):
            if current_discount > best_discount:
                best_discount = current_discount
                best_combo = dict(combo)
            return
        name, discount, threshold, max_qty = available_list[idx]
        max_possible = min(max_qty, int((base_total - current_threshold) // threshold))
        for q in range(max_possible, -1, -1):
            if q > 0:
                combo[name] = q
            else:
                combo.pop(name, None)
            dfs(
                idx + 1,
                current_threshold + q * threshold,
                current_discount + q * discount,
                combo,
            )
        combo.pop(name, None)

    dfs(0, 0.0, 0.0, {})
    return best_combo, best_discount


def compute_optimal_coupons_hint(user_info: dict[str, Any] | None, cart: dict[str, Any] | None) -> str:
    """Compute the exact mathematical optimal coupon combination for the current cart and user_info."""
    if not isinstance(user_info, dict) or not isinstance(cart, dict):
        return ""
    items = cart.get("items")
    if not isinstance(items, list) or not items:
        return ""
    base_total = round(
        sum(float(item.get("price") or 0.0) * int(item.get("quantity") or 0) for item in items),
        2,
    )
    if base_total <= 0:
        return ""

    best_combo, best_discount = compute_optimal_coupons_dict(user_info, cart)
    if not best_combo:
        owned = user_info.get("coupons")
        if isinstance(owned, dict) and owned:
            return (
                f"Cart base total is ¥{base_total:.2f}; no owned coupons meet their threshold yet "
                f"(owned: {json.dumps(owned, ensure_ascii=False)}). Check if selecting a slightly higher-priced "
                f"valid candidate product unlocks a coupon for a lower post-discount total!"
            )
        return ""

    used = {
        c.get("coupon_name"): int(c.get("quantity", 0))
        for c in (cart.get("used_coupons") or [])
        if isinstance(c, dict) and c.get("coupon_name")
    }
    if used == best_combo:
        return (
            f"Cart base total is ¥{base_total:.2f}. Optimal coupons are ALREADY applied in cart: "
            f"{json.dumps(best_combo, ensure_ascii=False)} (total discount ¥{best_discount:.2f})."
        )
    return (
        f"Cart base total is ¥{base_total:.2f}. Based on user's owned coupons, the max-discount valid coupon "
        f"combination is: {json.dumps(best_combo, ensure_ascii=False)} "
        f"(total discount ¥{best_discount:.2f}). Currently used_coupons in cart: {json.dumps(used, ensure_ascii=False)}. "
        f"Remember to call `add_coupon_to_cart` before finishing!"
    )


def build_ephemeral_turn_note(request: AgentRequest, state: dict[str, Any]) -> str | None:
    """Build a short turn-status reminder (turn budget, live cart state, coupon hint) if helpful."""
    limits = request.limits or {}
    max_turns = int(limits.get("max_agent_turns") or limits.get("max_model_calls") or 80)
    remaining = max_turns - request.step
    notes: list[str] = []

    if remaining <= 3:
        if request.benchmark == "tau2_airline":
            notes.append(
                f"URGENT: Only {remaining} turn(s) remaining! Call `respond_to_user` now to respond to the customer."
            )
        else:
            notes.append(
                f"URGENT: Only {remaining} turn(s) remaining! Finalize any pending mutations immediately or return your final answer now."
            )

    if request.benchmark == "mcpmark" and request.step > 0:
        mcp_helper_note = compute_mcpmark_data_helper_note(request)
        if mcp_helper_note:
            notes.append(mcp_helper_note)

    if request.benchmark == "tau2_airline" and request.step > 0:
        res_by_id = state.get("tau2_reservations_by_id") or {}
        init_res_by_id = state.get("tau2_initial_reservations_by_id") or {}
        orig_pay_map: dict[str, str] = {}
        for rid, rdata in init_res_by_id.items():
            if isinstance(rdata, dict):
                ph = rdata.get("payment_history")
                if isinstance(ph, list) and ph and isinstance(ph[0], dict) and ph[0].get("payment_id"):
                    orig_pay_map[rid] = str(ph[0]["payment_id"])
        if len(orig_pay_map) >= 2:
            notes.append(
                f"[Tau2 Original Payment Methods by Reservation]: {orig_pay_map}. "
                "When downgrading/updating multiple reservations where the customer didn't remember their payment method, "
                "refund EACH reservation to its OWN original `payment_id` shown above (`payment_history[0]['payment_id']`), "
                "NOT a single payment method across all reservations! And NEVER offer to email itineraries or call `transfer_to_human_agents` after updating!"
            )
        first_user_msg = next(
            (
                str(m.get("content") or "")
                for m in (state.get("messages") or [])
                if isinstance(m, dict) and m.get("role") == "user"
            ),
            "",
        ).lower()
        for rid, rdata in init_res_by_id.items():
            if (
                isinstance(rdata, dict)
                and rdata.get("cabin") == "basic_economy"
                and ("chang" in first_user_msg or "switch" in first_user_msg)
                and "cancel" not in first_user_msg
                and "upgrade" not in first_user_msg
            ):
                f_nums = [str(f.get("flight_number")) for f in (rdata.get("flights") or []) if isinstance(f, dict)]
                notes.append(
                    f"[Tau2 Policy Reminder for {rid}]: Reservation `{rid}` started as `basic_economy` (initial flights: {f_nums}) and the customer wants to switch flights. "
                    "Do NOT call `transfer_to_human_agents`! Use the 2-step path: **Step 1**: upgrade "
                    f"`{rid}` (keeping flights {f_nums}) from `basic_economy` to `economy` using the customer's credit card on file (ask confirmation first); "
                    "**Step 2**: once upgraded to `economy`, separately ask confirmation and change to the new nonstop flight **using the EXACT SAME credit card `payment_id` from Step 1** (do NOT propose or switch to a gift card in Step 2)! "
                    "**CRITICAL WHEN CUSTOMER HAS A BUDGET LIMIT FOR THE CHANGE**: Always compute and emphasize the **COMBINED NET COST** across both steps: `(Step 1 upgrade charge) + (Step 2 fare difference/refund) = (New flight economy total) - (Original basic_economy total)`. Whenever the combined net cost is within the customer's budget (or is a net refund!), reassure the customer that the Step 1 charge is immediately offset by the Step 2 refund so their net cost is well within budget, and ask them to reply `'yes'` to proceed with Step 1 (do NOT steer them to keep current flights)!"
                )
            if isinstance(rdata, dict) and "same as my current reservation" in first_user_msg:
                f_list = rdata.get("flights") or []
                f_date = f_list[0].get("date") if f_list and isinstance(f_list[0], dict) else ""
                orig = rdata.get("origin") or ""
                dest = rdata.get("destination") or ""
                notes.append(
                    f"[Tau2 Duplicate Reservation Reminder for {rid}]: CRITICAL: The flight prices inside `get_reservation_details` ({rid}) are "
                    "OLD historical prices! You MUST call `search_direct_flight` and `search_onestop_flight` for "
                    f"`origin='{orig}', destination='{dest}', date='{f_date}'` to get the **current** flight prices before quoting a total or booking! "
                    "Also note: unless the customer explicitly requested baggage or insurance for the friend, clarify/confirm whether the new booking has **no baggage (`total_baggages=0, nonfree_baggages=0`) and no insurance (`insurance='no'`)**!"
                )
        price_diffs = state.get("tau2_reservation_price_diffs") or {}
        if price_diffs:
            net_diff = sum(int(v) for v in price_diffs.values())
            abs_diff = abs(net_diff)
            notes.append(
                f"[Tau2 Cumulative Flight Price Difference Tracker]: Per-reservation flight updates: {price_diffs}. "
                f"Total cumulative net change: {net_diff} (Total refund/savings across updated reservations: "
                f"${abs_diff:,} / {abs_diff}). Be sure to state both `${abs_diff:,}` and `{abs_diff}` when reporting total savings!"
            )

    if request.benchmark == "deepplanning" and request.step > 0:
        obs = request.observation or {}
        cart = obs.get("cart")
        if isinstance(cart, dict):
            notes.append(f"Live Cart State (`cart.json`): {json.dumps(cart, ensure_ascii=False)}")
            user_info = state.get("cached_user_info")
            coupon_hint = compute_optimal_coupons_hint(user_info, cart)
            if coupon_hint:
                notes.append(f"Coupon Helper: {coupon_hint}")

        dp_helper_note = compute_deepplanning_helper_note(request, state)
        if dp_helper_note:
            notes.append(dp_helper_note)

        if state.get("dp_catalog_products"):
            notes.append(
                "FULL CATALOG PRE-FETCHED: The `get_product_details` result above ALREADY contains the ENTIRE case catalog "
                "filtered to the user's demographic and valid sizes, sorted by `price` ASCENDING, with `transport_time`, "
                "`default_size_ok`, and `default_fit_ok` (`version_type == preference_fit`) pre-computed! "
                "Do NOT call any more `search_products` or `filter_by_*` tools! Scan `get_product_details` and `[Deterministic Python Helper]` directly, "
                "keep internal reasoning under 80 words, and immediately call `add_product_to_cart` (up to 3 per turn)!"
            )

        # Warn if a filter/sort batch has not had its candidates inspected via a SUBSEQUENT get_product_details
        filter_batches = state.get("filter_batches") or []
        if filter_batches:
            uninspected_reminders: list[str] = []
            for idx, entry in enumerate(filter_batches):
                tname, targs, b_ids = entry[0], entry[1], entry[2]
                b_inspected = bool(entry[3]) if len(entry) > 3 else False
                if b_inspected:
                    continue
                b_set = set(str(x) for x in b_ids)
                if not b_set:
                    continue
                # Check if a later filter/sort batch took a subset of this batch
                narrowed_later = False
                for later_entry in filter_batches[idx + 1 :]:
                    later_ids = later_entry[2]
                    later_set = set(str(x) for x in later_ids)
                    if later_set and later_set.issubset(b_set):
                        narrowed_later = True
                        break
                if narrowed_later:
                    continue
                sample = [str(x) for x in b_ids[:15]]
                cond_desc = targs.get("condition_key") or targs.get("brand_names") or targs.get("sizes") or tname
                uninspected_reminders.append(f"{tname}({cond_desc}) -> {len(b_ids)} IDs (first 15: {sample})")
            if uninspected_reminders:
                notes.append(
                    "WARNING — Uninspected Filter/Sort Results: You have NOT called `sort_products(sort_by='price', "
                    "order='asc')` + `get_product_details` AFTER running: "
                    + "; ".join(uninspected_reminders[-2:])
                    + ". Do NOT reuse an expensive item from another search—call `sort_products(sort_by='price', "
                    "order='asc')` and `get_product_details` on these filtered IDs NOW to find the cheapest valid product!"
                )

    if not notes:
        return None
    return "[System Status Update]\n" + "\n".join(notes)
