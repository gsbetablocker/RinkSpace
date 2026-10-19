#include "rinkfit/augmented_lagrangian.hpp"

#include <iomanip>
#include <istream>
#include <locale>
#include <ostream>
#include <stdexcept>

namespace rinkfit {
namespace {
constexpr std::size_t kMaxSerializedItems = 20000000;
void check(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(reason);
}
std::size_t count(std::istream& in) {
  std::string token; in >> token;
  check(!token.empty() && token[0] != '-', "invalid checkpoint count");
  std::size_t used = 0;
  const auto result = std::stoull(token, &used);
  check(used == token.size() && result <= kMaxSerializedItems, "checkpoint count exceeds declared parser limit");
  return static_cast<std::size_t>(result);
}
double number(std::istream& in) {
  std::string token; in >> token;
  std::size_t used = 0;
  const double result = std::stod(token, &used);
  check(used == token.size() && !std::isnan(result), "invalid checkpoint number");
  return result;
}
void write_vector(std::ostream& out, const std::vector<double>& values) {
  out << values.size();
  for (double v : values) out << ' ' << v;
  out << '\n';
}
std::vector<double> read_vector(std::istream& in) {
  const std::size_t size = count(in);
  std::vector<double> values; values.reserve(size);
  for (std::size_t i = 0; i < size; ++i) values.push_back(number(in));
  return values;
}
void write_map(std::ostream& out, const std::map<std::string, std::string>& values) {
  out << values.size() << '\n';
  for (const auto& item : values) out << std::quoted(item.first) << ' ' << std::quoted(item.second) << '\n';
}
std::map<std::string, std::string> read_map(std::istream& in) {
  std::map<std::string, std::string> values;
  const std::size_t size = count(in);
  for (std::size_t i = 0; i < size; ++i) {
    std::string key, value; in >> std::quoted(key) >> std::quoted(value);
    check(values.emplace(key, value).second, "duplicate checkpoint discrete key");
  }
  return values;
}
void write_state(std::ostream& out, const StateSnapshot& state) {
  out << (!state.empty()) << '\n';
  if (state.empty()) return;
  const auto& data = state.data(); const auto& id = data.identity;
  out << std::quoted(state.token()) << '\n'
      << std::quoted(id.source_id) << ' ' << std::quoted(id.model_id) << ' '
      << std::quoted(id.budget_id) << ' ' << std::quoted(id.statistics_id) << ' '
      << std::quoted(id.scales_id) << ' ' << std::quoted(id.objective_id) << '\n'
      << data.parameters.size() << '\n';
  for (const auto& p : data.parameters) {
    out << std::quoted(p.name) << ' ' << p.equality_auxiliary << '\n';
    write_vector(out, p.values);
  }
  write_map(out, data.discrete); write_map(out, data.root_charts);
}
StateSnapshot read_state(std::istream& in) {
  const auto present = count(in); check(present <= 1, "invalid checkpoint snapshot flag");
  if (!present) return {};
  std::string token; in >> std::quoted(token);
  StatePayload data; auto& id = data.identity;
  in >> std::quoted(id.source_id) >> std::quoted(id.model_id) >> std::quoted(id.budget_id)
     >> std::quoted(id.statistics_id) >> std::quoted(id.scales_id) >> std::quoted(id.objective_id);
  const std::size_t blocks = count(in);
  std::size_t total_values = 0;
  for (std::size_t i = 0; i < blocks; ++i) {
    ParameterValues p; in >> std::quoted(p.name);
    const auto auxiliary = count(in); check(auxiliary <= 1, "invalid auxiliary flag");
    p.equality_auxiliary = auxiliary == 1; p.values = read_vector(in);
    total_values += p.values.size();
    check(total_values <= kMaxSerializedItems, "checkpoint total parameter size exceeds parser limit");
    data.parameters.push_back(std::move(p));
  }
  data.discrete = read_map(in); data.root_charts = read_map(in);
  StateSnapshot state(std::move(data));
  check(state.token() == token, "checkpoint snapshot token mismatch: corrupted or edited complete state");
  return state;
}
} // namespace

void write_checkpoint(std::ostream& out, const RunCheckpoint& checkpoint) {
  check(!checkpoint.state.empty(), "cannot save an empty current state");
  out.imbue(std::locale::classic()); out << std::setprecision(17);
  out << "RINKFIT_AL_CHECKPOINT_V1\n" << std::quoted(checkpoint.runtime_fingerprint) << '\n';
  const auto& a = checkpoint.outer;
  out << static_cast<int>(a.phase) << ' ' << a.outer_iteration << ' ' << a.inner_attempt << ' '
      << a.chart_restarts << ' ' << a.rho << ' ' << a.previous_progress << ' ' << a.elapsed_seconds << '\n';
  write_vector(out, a.lambda); write_vector(out, a.nu);
  write_state(out, checkpoint.state); write_state(out, checkpoint.best_physical_state);
  out << "END_RINKFIT_AL_CHECKPOINT\n";
  check(static_cast<bool>(out), "checkpoint write failed");
}
RunCheckpoint read_checkpoint(std::istream& in) {
  in.imbue(std::locale::classic());
  std::string magic; in >> magic;
  check(magic == "RINKFIT_AL_CHECKPOINT_V1", "unsupported checkpoint format");
  RunCheckpoint result;
  in >> std::quoted(result.runtime_fingerprint);
  const auto phase = count(in); check(phase == 1 || phase == 2, "invalid saved phase");
  auto& a = result.outer; a.phase = static_cast<Phase>(phase);
  a.outer_iteration = count(in); a.inner_attempt = count(in); a.chart_restarts = count(in);
  a.rho = number(in); a.previous_progress = number(in); a.elapsed_seconds = number(in);
  a.lambda = read_vector(in); a.nu = read_vector(in);
  result.state = read_state(in); result.best_physical_state = read_state(in);
  in >> magic; check(magic == "END_RINKFIT_AL_CHECKPOINT" && static_cast<bool>(in), "truncated checkpoint");
  check(!result.state.empty(), "checkpoint has no current state");
  // Full runtime/schema/manifest compatibility is checked by solve_two_phase.
  return result;
}
} // namespace rinkfit
