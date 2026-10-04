require "test_helper"

class EstimationResponseTest < ActiveSupport::TestCase
  test "from_hash builds result, tasks and summary with API field names" do
    response = EstimationResponse.from_hash(estimation_payload(cached: true))

    assert_equal "v1", response.prompt_version
    assert response.cached

    result = response.result
    assert_kind_of EstimationResult, result
    assert_equal "Customer Portal & Invoice Management System", result.project_name
    assert_predicate result.meeting_summary, :present?

    task = result.tasks.first
    assert_kind_of Task, task
    assert_equal 1, task.task_id
    assert_equal "Medium", task.complexity
    assert_equal 18, task.estimated_hours
    assert_equal 900, task.estimated_cost_usd
    assert_includes task.includes, "Password reset flow"

    summary = result.summary
    assert_kind_of EstimationSummary, summary
    assert_equal 298, summary.total_hours
    assert_equal 14_900, summary.total_cost_usd
    assert_equal 7, summary.estimated_duration_weeks
    assert_equal 50, summary.hourly_rate
    assert_match(/2 developers/, summary.team_size)
    assert_equal 11, summary.assumptions.size
  end

  test "cached defaults to false and assumptions to empty" do
    payload = estimation_payload.tap { |p| p.delete("cached"); p["result"]["summary"].delete("assumptions") }
    response = EstimationResponse.from_hash(payload)

    assert_equal false, response.cached
    assert_equal [], response.result.summary.assumptions
  end
end
