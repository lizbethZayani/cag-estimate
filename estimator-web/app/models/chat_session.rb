# Local mirror of a conversation held by the estimator API. The API owns the
# conversation memory; this row keeps the remote id, the latest metadata
# snapshot and links the estimations produced in that conversation.
class ChatSession < ApplicationRecord
  METADATA_DEFAULTS = {
    "project_name" => nil,
    "assumed_team_size" => nil,
    "mentioned_technologies" => [],
    "agreed_scope" => nil
  }.freeze

  has_many :estimations, dependent: :nullify

  validates :remote_session_id, presence: true, uniqueness: true

  def latest_metadata_hash
    stored = latest_metadata.is_a?(Hash) ? latest_metadata.stringify_keys : {}
    METADATA_DEFAULTS.deep_dup.merge(stored) do |_key, default, value|
      value.nil? ? default : value
    end
  end

  def title
    latest_metadata_hash["project_name"].presence || "Untitled conversation"
  end
end
