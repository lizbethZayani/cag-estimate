class EstimationResult
  include ActiveModel::Model
  include ActiveModel::Attributes

  attribute :project_name, :string
  attribute :meeting_summary, :string

  attr_reader :tasks, :summary

  def initialize(attributes = {})
    stringified = attributes.to_h.transform_keys(&:to_s)
    tasks_data = stringified.delete("tasks") || []
    summary_data = stringified.delete("summary") || {}
    super(stringified.slice("project_name", "meeting_summary"))
    @tasks = tasks_data.map { |raw| Task.new(raw.to_h.transform_keys(&:to_s).slice(*Task.attribute_names)) }
    @summary = EstimationSummary.new(
      summary_data.to_h.transform_keys(&:to_s).slice(*EstimationSummary.attribute_names)
    )
  end
end
